from dataclasses import dataclass, field
import math
from typing import List

import torch

from byprot.models import register_model
from byprot.models.fixedbb import FixedBackboneDesignEncoderDecoder
from byprot.models.fixedbb.generator import sample_from_categorical
from byprot.models.fixedbb.protein_mpnn_cmlm.protein_mpnn import ProteinMPNNCMLM, ProteinMPNNConfig

from .modules.esm2_adapter import ESM2WithStructuralAdatper


@dataclass
class ESM2AdapterConfig:
    encoder: ProteinMPNNConfig = field(default=ProteinMPNNConfig())
    adapter_layer_indices: List = field(default_factory=lambda: [32, ])
    separate_loss: bool = True
    name: str = 'esm2_t33_650M_UR50D'
    dropout: float = 0.1
    # ensemble_logits: bool = False


@register_model('esm2_adapter')
class ESM2Adapter(FixedBackboneDesignEncoderDecoder):
    _default_cfg = ESM2AdapterConfig()

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.encoder = ProteinMPNNCMLM(self.cfg.encoder)
        self.decoder = ESM2WithStructuralAdatper.from_pretrained(args=self.cfg, name=self.cfg.name)

        self.padding_idx = self.decoder.padding_idx
        self.mask_idx = self.decoder.mask_idx
        self.cls_idx = self.decoder.cls_idx
        self.eos_idx = self.decoder.eos_idx

    def forward(self, batch, **kwargs):
        encoder_logits, encoder_out = self.encoder(batch, return_feats=True, **kwargs)

        encoder_out['feats'] = encoder_out['feats'].detach()

        init_pred = encoder_logits.argmax(-1)
        init_pred = torch.where(batch['coord_mask'], init_pred, batch['prev_tokens'])

        esm_logits = self.decoder(
            tokens=init_pred,
            encoder_out=encoder_out,
        )['logits']

        if not getattr(self.cfg, 'separate_loss', False):
            logits = encoder_logits + esm_logits
            return logits, encoder_logits
        else:
            return esm_logits, encoder_logits

    def forward_encoder(self, batch):
        encoder_logits, encoder_out = self.encoder(batch, return_feats=True)

        init_pred = encoder_logits.argmax(-1)
        init_pred = torch.where(batch['coord_mask'], init_pred, batch['prev_tokens'])

        encoder_out['logits'] = encoder_logits
        encoder_out['init_pred'] = init_pred
        encoder_out['coord_mask'] = batch['coord_mask']
        return encoder_out

    def forward_decoder(self, prev_decoder_out, encoder_out, need_attn_weights=False):
        output_tokens = prev_decoder_out['output_tokens']
        output_scores = prev_decoder_out['output_scores']
        step, max_step = prev_decoder_out['step'], prev_decoder_out['max_step']
        temperature = prev_decoder_out['temperature']
        history = prev_decoder_out['history']
        tied_pos_list = prev_decoder_out['tied_pos_list']

        # output_masks = output_tokens.eq(self.mask_idx)  # & coord_mask
        output_masks = output_tokens.ne(self.padding_idx)  # & coord_mask

        esm_out = self.decoder(
            tokens=output_tokens,
            encoder_out=encoder_out,
            need_head_weights=need_attn_weights
        )
        esm_logits = esm_out['logits']
        attentions = esm_out['attentions'] if need_attn_weights else None

        if not getattr(self.cfg, 'separate_loss', False):
            logits = esm_logits + encoder_out['logits']
        else:
            logits = esm_logits  # + encoder_out['logits']
        logits[..., self.mask_idx] = -math.inf
        logits[..., self.decoder.alphabet.get_idx("X")] = -math.inf

        # if tied positions are given, 
        # we force the logits of the tied positions
        # to be their sumation
        logits = apply_tied_pos_fn(tied_pos_list, logits, fn=lambda x: x.sum(0))

        _tokens, _scores = sample_from_categorical(logits, temperature=temperature)

        # if tied positions are given,
        # we also force the prediction of the tied positions
        # to be the same
        _tokens = apply_tied_pos_fn(tied_pos_list, _tokens, fn=lambda x: x[0])
        _scores = apply_tied_pos_fn(tied_pos_list, _scores, fn=lambda x: x[0])

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        history.append(output_tokens.clone())

        return dict(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attentions=attentions, # [B, L, H, T, T]
            step=step + 1,
            max_step=max_step,
            history=history,
        )

    def initialize_output_tokens(self, batch, encoder_out):
        mask = encoder_out.get('coord_mask', None)

        prev_tokens = batch['prev_tokens']
        prev_token_mask = batch['prev_token_mask']
        # lengths = prev_tokens.ne(self.padding_idx).sum(1)

        # initial_output_tokens = torch.full_like(prev_tokens, self.padding_idx)
        # initial_output_tokens.masked_fill_(new_arange(prev_tokens) < lengths[:, None], self.mask_idx)
        # initial_output_tokens[:, 0] = self.cls_idx
        # initial_output_tokens.scatter_(1, lengths[:, None] - 1, self.eos_idx)

        # initial_output_tokens = encoder_out['init_pred'].clone()
        initial_output_tokens = torch.where(
            prev_token_mask, encoder_out['init_pred'], prev_tokens)
        initial_output_scores = torch.zeros(
            *initial_output_tokens.size(), device=initial_output_tokens.device
        )

        return initial_output_tokens, initial_output_scores

    def sample(
        self, 
        batch, 
        max_iter=5, 
        strategy="denoise", 
        temperature=None, 
        replace_visible_tokens=True, 
        need_attn_weights=False
    ):         
        encoder_out = self.forward_encoder(batch)

        # 1) initialized from all mask tokens
        initial_output_tokens, initial_output_scores = self.initialize_output_tokens(
            batch, encoder_out=encoder_out
        )
        prev_decoder_out = dict(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            output_masks=None,
            attentions=None,
            step=0,
            max_step=max_iter,
            history=[initial_output_tokens.clone()],
            temperature=temperature,
            tied_pos_list=batch["tied_pos_list"]
        )

        if need_attn_weights:
            attns = [] # list of {'in', 'out', 'attn'} for all iteration

        if strategy == 'discrete_diffusion':
            prev_decoder_out['output_masks'] = self.get_non_special_sym_mask(batch['prev_tokens'])

        # iterative refinement
        for step in range(max_iter):

            # 2.1: predict
            decoder_out = self.forward_decoder(
                prev_decoder_out=prev_decoder_out,
                encoder_out=encoder_out,
                need_attn_weights=need_attn_weights
            )

            output_tokens = decoder_out['output_tokens']
            output_scores = decoder_out['output_scores']

            # 2.2: re-mask skeptical parts of low confidence
            # skeptical decoding (depend on the maximum decoding steps.)
            if (
                strategy == 'mask_predict'
                and (step + 1) < max_iter
            ):
                skeptical_mask = _skeptical_unmasking(
                    output_scores=output_scores,
                    output_masks=output_tokens.ne(self.padding_idx),  # & coord_mask,
                    p=1 - (step + 1) / max_iter
                )

                output_tokens.masked_fill_(skeptical_mask, self.mask_idx)
                output_scores.masked_fill_(skeptical_mask, 0.0)

            elif strategy == 'denoise' or strategy == 'no':
                pass
            elif strategy == 'discrete_diffusion':
                pass
            else:
                pass

            if replace_visible_tokens:
                visible_token_mask = ~batch['prev_token_mask']
                visible_tokens = batch['prev_tokens']
                output_tokens = torch.where(
                    visible_token_mask, visible_tokens, output_tokens)

            if need_attn_weights:
                attns.append(
                    dict(input=maybe_remove_batch_dim(prev_decoder_out['output_tokens']),
                         output=maybe_remove_batch_dim(output_tokens),
                         attn_weights=maybe_remove_batch_dim(decoder_out['attentions']))
                )

            prev_decoder_out.update(
                output_tokens=output_tokens,
                output_scores=output_scores,
                step=step + 1,
                history=decoder_out['history']
            )

        decoder_out = prev_decoder_out

        if need_attn_weights:
            return decoder_out['output_tokens'], decoder_out['output_scores'], attns
        return decoder_out['output_tokens'], decoder_out['output_scores']


def exists(obj):
    return obj is not None


def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


def maybe_remove_batch_dim(tensor):
    if len(tensor.shape) > 1 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor


def _skeptical_unmasking(output_scores, output_masks, p):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = ((output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * p).long()
    # `length * p`` positions with lowest scores get kept
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)

def apply_tied_pos_fn(tied_pos_list, data, fn):
    if not exists(tied_pos_list):
       return data 

    for bid, tied_pos in enumerate(tied_pos_list):
        for tp in tied_pos:
            data[bid, tp] = fn(data[bid, tp])
    return data
