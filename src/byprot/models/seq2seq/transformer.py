import contextlib
from functools import partial

from torch import nn
from torch.nn import functional as F

from .modules.embedding import Embedding, PositionEmbedding


def bert_init_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()

    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()

    # if isinstance(module, MultiheadAttention):
    #     normal_(module.q_proj.weight.data)
    #     normal_(module.k_proj.weight.data)
    #     normal_(module.v_proj.weight.data)


def fairseq_init_params(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)

    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=module.embedding_dim ** -0.5)
        if module.padding_idx is not None:
            nn.init.constant_(module.weight[module.padding_idx], 0.0)


def create_padding_mask(x, pad=1):
    return x.ne(pad)


def _set_inferring_flag(mod, flag=True):
    mod._inferring = flag


class TransformerEncoderDecoder(nn.Module):
    def __init__(
        self,
        cfg,
        embed_src=None,
        embed_tgt=None,
        pos_embed_src=None,
        pos_embed_tgt=None,
        encoder=None,
        decoder=None,
        out_proj=None,
    ):
        super().__init__()

        self.cfg = cfg
        self.d_model = self.cfg.d_model

        self.embed_src, self.embed_tgt = embed_src, embed_tgt
        self.out_proj = out_proj

        self.pos_embed_src, self.pos_embed_tgt = pos_embed_src, pos_embed_tgt

        self.encoder = encoder
        self.decoder = decoder

        self.reset_parameters()

    @classmethod
    def build(
        cls, cfg,
        vocab_src, vocab_tgt,
        token_embedding, position_embedding,
        encoder, decoder,
    ) -> "TransformerEncoderDecoder":

        src_embed = token_embedding(len(vocab_src), cfg.d_model, padding_idx=vocab_src.pad)
        tgt_embed = token_embedding(len(vocab_tgt), cfg.d_model, padding_idx=vocab_tgt.pad)
        out_proj = nn.Linear(cfg.d_model, len(vocab_tgt), bias=False)

        src_pos_embed = position_embedding(cfg.d_model)
        tgt_pos_embed = position_embedding(cfg.d_model)

        model = cls(cfg, src_embed, tgt_embed,
                    src_pos_embed, tgt_pos_embed, encoder, decoder, out_proj)
        model.apply(partial(_set_inferring_flag, flag=False))

        # initializtion model with BERT style, which was found magically good...
        model.apply(bert_init_params)

        if cfg.share_input_output_embedding:
            out_proj.weight = tgt_embed.weight
        if cfg.share_source_target_embedding:
            src_embed.weight = tgt_embed.weight
        return model

    def reset_parameters(self):
        for child in self.children():
            child.reset_parameters()

    @contextlib.contextmanager
    def inference_mode(self, mode=True):
        self.apply(partial(_set_inferring_flag, flag=mode))
        yield
        self.apply(partial(_set_inferring_flag, flag=False))

    def forward(self, src_tokens, tgt_tokens, src_padding_mask, tgt_padding_mask):
        """
        Args:
            src_tokens (LongTensor): source tokens [bsz, slen]
            tgt_tokens (LongTensor): target tokens [bsz, tlen]
        """
        encoder_out = self.encode(src_tokens, src_padding_mask)
        decoder_out = self.decode(tgt_tokens, encoder_out, tgt_padding_mask, src_padding_mask)
        logits = self.output(decoder_out, normalize=False)
        return logits

    def encode(self, src_tokens, src_padding_mask=None):
        src_emb = self.embed_src(src_tokens)
        encoder_input = self.pos_embed_src(src_emb)
        encoder_out = self.encoder(encoder_input, src_padding_mask)
        return encoder_out

    def decode(self, tgt_tokens, encoder_out,
               tgt_padding_mask=None, src_padding_mask=None,
               incremental_states=None):
        tgt_emb = self.embed_tgt(tgt_tokens)
        decoder_input = self.pos_embed_tgt(tgt_emb)

        if incremental_states is not None:
            # [bsz, len, d] -> [bsz, 1, d]
            decoder_input = decoder_input[:, -1:]
            tgt_padding_mask = None

        decoder_out = self.decoder(decoder_input, encoder_out,
                                   tgt_padding_mask, src_padding_mask,
                                   incremental_states=incremental_states)
        return decoder_out

    def output(self, decoder_out, normalize=True):
        logits = self.out_proj(decoder_out)
        return F.log_softmax(logits, dim=-1) if normalize else logits
