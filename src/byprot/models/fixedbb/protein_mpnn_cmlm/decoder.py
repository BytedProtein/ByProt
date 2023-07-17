from typing import Dict
import torch
from torch import nn
import numpy as np

from .features import ProteinFeatures, gather_nodes, cat_neighbors_nodes, PositionWiseFeedForward

def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


def _skeptical_unmasking(output_scores, output_masks, p):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = (
        (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * p
    ).long()
    # `length * p`` positions with lowest scores get kept
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)


def exists(obj):
    return obj is not None


def get_neighbors(self, X, mask, top_k=5, eps=1E-6):
    mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
    dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
    D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
    D_max, _ = torch.max(D, -1, keepdim=True)
    D_adjust = D + (1. - mask_2D) * D_max
    D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(top_k, X.shape[1]), dim=-1, largest=False)
    return D_neighbors, E_idx


def convert_neighbors_to_binary_edges(edge_idx):
    # edge_idx: [B, N, K]
    # return: [B, N*K, 2]
    B, N, K = edge_idx.shape
    start_nodes = torch.arange(N, device=edge_idx.device)[None, :, None].expand(B, N, K)  # [B, N, K]
    end_nodes = edge_idx  # [B, N, K]
    binary_edges = torch.stack([start_nodes, end_nodes], dim=-1)
    binary_edges = binary_edges.reshape(B, -1, 2)
    binary_masks = torch.ones(*binary_edges.shape[:2], device=edge_idx.device)
    return binary_edges, binary_masks


class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        # h_message: [bsz, n_nodes, n_edges, d_nodes + d_edges]
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))

        # mask_attend: [bsz, n_nodes, n_edges]
        if mask_attend is not None:
            h_message = mask_attend[:, :, :, None] * h_message
        # dh: aggregated neighbors' features, [bsz, n_nodes, d]
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            # mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V[:, :, None] * h_V
        # final node features: [bsz, n_nodes, d]
        return h_V


class MPNNDecoder(nn.Module):
    def __init__(
        self, hidden_dim, num_decoder_layers=3,
        vocab=22, k_neighbors=64, dropout=0.1,
        random_decoding_order=True, nar=False,
        token_embed=None, out_proj=None
    ):
        super().__init__()

        self.token_embed = token_embed or nn.Embedding(vocab, hidden_dim)
        self.out_proj = out_proj or nn.Linear(hidden_dim, vocab, bias=True)

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])

        self.random_decoding_order = random_decoding_order
        self.nar = nar

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _prepare_decoding_order(self, mask, E_idx, chain_mask=None, decoding_order=None):
        device = mask.device

        if chain_mask is None:
            chain_mask = torch.ones((E_idx.shape[0], E_idx.shape[1])).to(mask.device)
        chain_mask = chain_mask * mask  # update chain_M to include missing regions

        if decoding_order is None:
            if self.random_decoding_order:
                randn = torch.randn(chain_mask.shape, device=device)
            else:
                # deterministic left-to-right
                decoding_order = torch.arange(0, chain_mask.shape[1])[None, :].repeat(chain_mask.shape[0], 1).to(device)
            # [numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
            # i.e., masked positions get small values and and ranked lower, unmaksed one get ranked higher
            decoding_order = torch.argsort((chain_mask + 0.0001) * (torch.abs(randn)))

        mask_size = E_idx.shape[1]

        if not self.nar:
            # [bsz, n_nodes, n_nodes]
            permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
            order_mask_backward = torch.einsum(
                'ij, biq, bjp->bqp',
                1 - torch.triu(torch.ones(mask_size, mask_size, device=device)),  # low triangular without diagonal, the AR mask
                permutation_matrix_reverse,
                permutation_matrix_reverse
            )
            # permuted autoregressive mask with regards to edge/neighbors
            # for a node, which edges can be attended to
            mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
            mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
            # for a node i, mask_bw[i] + mask_fw[i] == n_edges
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1. - mask_attend)
        else:
            mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
            mask_attend = torch.zeros_like(E_idx).to(mask_1D).unsqueeze(-1)
            # mask_bw = mask_1D * mask_attend
            mask_bw = mask_1D * (1. - mask_attend)
            mask_fw = mask_1D * (1. - mask_attend)

        return mask_fw, mask_bw, decoding_order, chain_mask

    def _preprocess_structure_feats(self, h_S, h_V, h_E, E_idx, mask):
        # Concatenate sequence embeddings with structure features
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        return h_ES, h_EXV_encoder

    def forward(self, h_S, h_V, h_E, E_idx, mask, chain_mask=None, decoding_order=None):
        # get causal masks if random decoding order
        mask_fw, mask_bw, deocding_order, chain_mask = self._prepare_decoding_order(
            mask, E_idx, decoding_order=decoding_order, chain_mask=chain_mask,
        )

        # get preprocessed structure features
        h_ES, h_EXV_encoder = self._preprocess_structure_feats(
            h_S, h_V, h_E, E_idx, mask
        )

        # h_EXV_encoder: [bsz, n_nodes, n_edges, d_edges],
        # node's neighbor features from graph encoder, without sequence informed
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            # h_ESV: [bsz, n_nodes, n_edges, d_edges], node's neighbor features with sequence informed
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)

            # Unmaksed positions attend to structure-sequence information
            # Masked positions only attend to structure information (from encoder)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw

            h_V = layer(h_V, h_ESV, mask)

        return h_V

    def sample_nar(self, h_S, h_V, h_E, E_idx, mask, chain_mask=None, decoding_order=None, temperature=1e-2):
        device = mask.device

        h_V_t = self.forward(h_S, h_V, h_E, E_idx, mask)

        logits = self.out_proj(h_V_t)
        return logits.argmax(-1)

    def sample(self, h_V, h_E, E_idx, mask, h_S=None, chain_mask=None, decoding_order=None, temperature=1e-2):
        if self.nar:
            return self.sample_nar(
                h_S, h_V, h_E, E_idx, mask,
                chain_mask=chain_mask, decoding_order=decoding_order, temperature=temperature
            )
        device = mask.device
        N_batch, N_nodes = mask.shape[0], mask.shape[1]

        # get causal masks if random decoding order
        mask_fw, mask_bw, decoding_order, chain_mask = self._prepare_decoding_order(
            mask, E_idx, decoding_order=decoding_order, chain_mask=chain_mask
        )

        # log_probs = torch.zeros((N_batch, N_nodes, 22), device=device)
        all_probs = torch.zeros((N_batch, N_nodes, 22), device=device, dtype=torch.float32)
        h_S = torch.zeros_like(h_V, device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(len(self.decoder_layers))]

        # constant, constant_bias = torch.zeros_like(all_probs)
        # constant = torch.tensor(omit_AAs_np, device=device)
        # constant_bias = torch.tensor(bias_AAs_np, device=device)
        # chain_mask_combined = chain_mask*chain_M_pos
        # omit_AA_mask_flag = omit_AA_mask != None

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder

        for t_ in range(N_nodes):
            t = decoding_order[:, t_]  # [B]
            chain_mask_gathered = torch.gather(chain_mask, 1, t[:, None])  # [B]

            # Hidden layers
            E_idx_t = torch.gather(E_idx, 1, t[:, None, None].repeat(1, 1, E_idx.shape[-1]))
            h_E_t = torch.gather(h_E, 1, t[:, None, None, None].repeat(1, 1, h_E.shape[-2], h_E.shape[-1]))
            h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
            h_EXV_encoder_t = torch.gather(
                h_EXV_encoder_fw,
                1, t[:, None, None, None].repeat(1, 1, h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1])
            )
            mask_t = torch.gather(mask, 1, t[:, None])
            for l, layer in enumerate(self.decoder_layers):
                # Updated relational features for future states
                h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                h_V_t = torch.gather(
                    h_V_stack[l],
                    1, t[:, None, None].repeat(1, 1, h_V_stack[l].shape[-1]))
                h_ESV_t = torch.gather(
                    mask_bw,
                    1, t[:, None, None, None].repeat(1, 1, mask_bw.shape[-2], mask_bw.shape[-1])
                ) * h_ESV_decoder_t + h_EXV_encoder_t
                h_V_stack[l + 1].scatter_(
                    1, t[:, None, None].repeat(1, 1, h_V.shape[-1]),
                    layer(h_V_t, h_ESV_t, mask_V=mask_t)  # layer update
                )

            # Sampling step
            h_V_t = torch.gather(
                h_V_stack[-1],
                1, t[:, None, None].repeat(1, 1, h_V_stack[-1].shape[-1])
            )[:, 0]
            logits = self.out_proj(h_V_t)
            probs = F.softmax(logits / temperature, dim=-1)
            # probs = F.softmax(
            #     logits - constant[None, :] * 1e8 + constant_bias[None, :] / temperature,
            #     dim=-1
            # )

            S_t = torch.multinomial(probs, 1)
            all_probs.scatter_(
                1, t[:, None, None].repeat(1, 1, 22),
                (chain_mask_gathered[:, :, None, ] * probs[:, None, :]).float()
            )

            # S_true_gathered = torch.gather(S_true, 1, t[:, None])
            # S_t = (S_t * chain_mask_gathered + S_true_gathered * (1.0 - chain_mask_gathered)).long()
            S_t = (S_t * chain_mask_gathered).long()
            S.scatter_(1, t[:, None], S_t)
            h_S_t = self.token_embed(S_t)
            h_S.scatter_(
                1, t[:, None, None].repeat(1, 1, h_S_t.shape[-1]),
                h_S_t
            )
        output_dict = {"S": S, "probs": all_probs, "decoding_order": decoding_order}
        return S


class SequenceDecoder(nn.Module):
    def __init__(self, n_vocab, d_model, **kwargs) -> None:
        super().__init__()

    def forward(
        self,
        prev_tokens: torch.FloatTensor,
        prev_token_padding_mask: torch.BoolTensor,
        memory: torch.FloatTensor = None,
        memory_padding_mask: torch.BoolTensor = None,
        **kwargs
    ):
        raise NotImplementedError


class LinearSequenceDecoder(SequenceDecoder):
    def __init__(self, n_vocab, d_model) -> None:
        super().__init__(n_vocab, d_model)

        self.out_proj = nn.Linear(d_model, n_vocab, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        prev_tokens: torch.FloatTensor,
        prev_token_padding_mask: torch.BoolTensor,
        memory: torch.FloatTensor = None,
        memory_padding_mask: torch.BoolTensor = None,
        **kwargs
    ):
        logits = self.out_proj(prev_tokens)
        return logits


class MPNNSequenceDecoder(SequenceDecoder):
    def __init__(
        self, n_vocab, d_model, n_layers,
        dropout=0.1, random_decoding_order=True,
        nar=False, crf=False,
        alphabet=None, token_embed=None, out_proj=None
    ) -> None:
        super().__init__(n_vocab, d_model)

        if alphabet is not None:
            n_vocab = len(alphabet)
            self.pad = alphabet.padding_idx
            self.unk = alphabet.unk_idx
        else:
            self.pad = 0
            self.unk = 1

        if token_embed is not None:
            self.token_embed = token_embed
        else:
            self.token_embed = nn.Embedding(n_vocab, d_model)

        if out_proj is not None:
            self.out_proj = out_proj
        else:
            self.out_proj = nn.Linear(d_model, n_vocab, bias=True)

        self.nar = nar
        self.mpnn_decoder = MPNNDecoder(
            hidden_dim=d_model, num_decoder_layers=n_layers,
            random_decoding_order=random_decoding_order, dropout=dropout,
            token_embed=self.token_embed, out_proj=self.out_proj,
            nar=nar
        )

        if crf:
            from torch_random_fields.models import GeneralCRF
            from torch_random_fields.models.constants import Training, Inference
            self.crf = GeneralCRF(
                num_states=n_vocab,
                feature_size=d_model,
                beam_size=64,
                low_rank=n_vocab,
                training=Training.PIECEWISE,
                inference=Inference.BATCH_BELIEF_PROPAGATION
            )
        else:
            self.crf = None

    def run_crf(self, logits, masks, edge_idx, node_features=None, max_binary_edges=None, target_tokens=None, decoding=False):
        # max_binary_edges = 5
        if max_binary_edges:  # cut-off neighbors for (possibly) more accurate PGM estimate
            edge_idx = edge_idx[:, :max_binary_edges]

        binary_edges, binary_masks = convert_neighbors_to_binary_edges(edge_idx)

        if not decoding:
            crf_loss = self.crf(
                unaries=logits,
                masks=masks,
                binary_edges=binary_edges,
                binary_masks=binary_masks.to(logits),
                targets=target_tokens,
                node_features=node_features,
            )
            return crf_loss
        else:
            output_tokens = self.crf(
                unaries=logits,
                masks=masks,
                binary_edges=binary_edges,
                binary_masks=binary_masks.to(logits),
                node_features=node_features,
            )[1]
            return output_tokens

    def forward(
        self,
        prev_tokens: torch.LongTensor,
        prev_token_padding_mask: torch.BoolTensor = None,
        memory: torch.FloatTensor = None,
        memory_mask: torch.FloatTensor = None,
        memory_padding_mask: torch.BoolTensor = None,
        target_tokens: torch.LongTensor = None,
        chain_mask=None,
        **kwargs
    ):
        h_tokens = self.token_embed(prev_tokens)
        node_feats, edge_feats, edge_idx = memory['node_feats'], memory['edge_feats'], memory['edge_idx']

        if memory_mask is None and memory_padding_mask is not None:
            memory_mask = (~memory_padding_mask).float()

        feats = self.mpnn_decoder(
            h_S=h_tokens, h_V=node_feats, h_E=edge_feats,
            E_idx=edge_idx, mask=memory_mask, chain_mask=chain_mask
        )

        logits = self.out_proj(feats)

        if exists(self.crf):
            crf_loss = self.run_crf(
                logits=logits,
                masks=memory_mask,
                edge_idx=edge_idx,
                node_features=feats,
                target_tokens=target_tokens,
            )
            return logits, crf_loss
        return logits, {'feats': feats, **memory}

    def sample(
        self,
        prev_tokens: torch.LongTensor = None,
        prev_token_padding_mask: torch.BoolTensor = None,
        memory: torch.FloatTensor = None,
        memory_mask: torch.FloatTensor = None,
        memory_padding_mask: torch.BoolTensor = None,
        **kwargs
    ):
        if self.nar:
            return self.sample_nar(prev_tokens, prev_token_padding_mask, memory, memory_mask, memory_padding_mask, **kwargs)
        else:
            return self.sample_ar(prev_tokens, prev_token_padding_mask, memory, memory_mask, memory_padding_mask, **kwargs)

    def sample_nar(
        self,
        prev_tokens: torch.LongTensor = None,
        prev_token_padding_mask: torch.BoolTensor = None,
        memory: torch.FloatTensor = None,
        memory_mask: torch.FloatTensor = None,
        memory_padding_mask: torch.BoolTensor = None,
        **kwargs
    ):
        output_tokens, output_scores = self.initialize_output_tokens(prev_tokens)

        node_feats, edge_feats, edge_idx = memory['node_feats'], memory['edge_feats'], memory['edge_idx']

        if memory_mask is None and memory_padding_mask is not None:
            memory_mask = (~memory_padding_mask).float()

        history = [output_tokens.clone()]
        max_step = 1

        coord_mask = memory_mask.bool()
        for step in range(max_step):
            output_masks = output_tokens.eq(self.unk)  # & coord_mask

            h_tokens = self.token_embed(output_tokens)
            decoder_out = self.mpnn_decoder(
                h_S=h_tokens,
                h_V=node_feats.clone(), h_E=edge_feats.clone(), E_idx=edge_idx.clone(), mask=memory_mask.clone()
            )

            if exists(self.crf):
                _tokens = self.run_crf(
                    logits=self.out_proj(decoder_out),
                    masks=memory_mask.bool(),
                    edge_idx=edge_idx,
                    node_features=decoder_out,
                    decoding=True
                )
                _scores = output_scores.clone()
            else:
                log_probs = torch.log_softmax(
                    self.out_proj(decoder_out),
                    dim=-1
                )
                _scores, _tokens = log_probs.max(-1)

            output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
            output_scores.masked_scatter_(output_masks, _scores[output_masks])

            # print(f"{step} [origin]: {output_tokens[0]}")
            # print(f"{step} [origin]: {output_scores[0]}")
            # skeptical decoding (depend on the maximum decoding steps.)
            if (step + 1) < max_step:
                skeptical_mask = _skeptical_unmasking(
                    output_scores=output_scores,
                    output_masks=output_tokens.ne(self.pad),  # & coord_mask,
                    p=1 - (step + 1) / max_step
                )

                output_tokens.masked_fill_(skeptical_mask, self.unk)
                output_scores.masked_fill_(skeptical_mask, 0.0)
            # print(f"{step} [maksed]: {output_scores[0]}")
            # print(f"{step} [masked]: {output_tokens[0]}")

            history.append(output_tokens.clone())

        # skeptical_mask = _skeptical_unmasking(
        #     output_scores=output_scores,
        #     output_masks=output_tokens.ne(self.pad),
        #     p=0.05
        # )

        # output_tokens.masked_fill_(skeptical_mask, self.unk)
        # output_scores.masked_fill_(skeptical_mask, 0.0)
        # print(f"GT: {prev_tokens[0]}")
        return output_tokens

    def initialize_output_tokens(self, prev_tokens, mask=None):
        lengths = prev_tokens.ne(self.pad).sum(1)

        initial_output_tokens = torch.full_like(prev_tokens, self.pad)
        initial_output_tokens.masked_fill_(new_arange(prev_tokens) < lengths[:, None], self.unk)

        if mask is not None:
            initial_output_tokens = torch.where(
                ~mask, prev_tokens, initial_output_tokens
            )

        # initial_output_tokens = prev_tokens.clone()

        initial_output_scores = torch.zeros(
            *initial_output_tokens.size(), device=initial_output_tokens.device
        )

        return initial_output_tokens, initial_output_scores

    def sample_ar(
        self,
        prev_tokens: torch.LongTensor = None,
        prev_token_padding_mask: torch.BoolTensor = None,
        memory: torch.FloatTensor = None,
        memory_mask: torch.FloatTensor = None,
        memory_padding_mask: torch.BoolTensor = None,
        **kwargs
    ):
        # initial_tokens = self.initialize_sampling(prev_tokens)

        # h_tokens = self.token_embed(initial_tokens)
        node_feats, edge_feats, edge_idx = memory['node_feats'], memory['edge_feats'], memory['edge_idx']

        if memory_mask is None and memory_padding_mask is not None:
            memory_mask = (~memory_padding_mask).float()

        preds = self.mpnn_decoder.sample(
            h_V=node_feats, h_E=edge_feats, E_idx=edge_idx, mask=memory_mask
        )
        return preds
