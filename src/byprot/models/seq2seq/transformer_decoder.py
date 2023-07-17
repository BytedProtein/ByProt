""" Transformer encoder """

import copy
from typing import Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .modules.ffn import FFN
from .modules.multihead_attention import MHA
from .modules.utils import ResNorm, _get_clones


class TransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            d_model,
            n_heads,
            d_inner=2048,
            dropout=0.1,
            attn_dropout=0.,
            normalize_before=False,
        ):
        super().__init__()

        self.self_attn = ResNorm(
            net=MHA(embed_dim=d_model, num_heads=n_heads, dropout=attn_dropout),
            fn=lambda net, x, *args, **kwargs: net(query=x, key=x, value=x, *args, **kwargs),
            dim=d_model, normalize_before=normalize_before, dropout=dropout
        )

        self.cross_attn = ResNorm(
            net=MHA(embed_dim=d_model, num_heads=n_heads, dropout=attn_dropout),
            dim=d_model, normalize_before=normalize_before, dropout=dropout
        )

        self.ffn = ResNorm(
            net=FFN(d_model=d_model, d_inner=d_inner, dropout=dropout),
            dim=d_model, normalize_before=normalize_before, dropout=dropout
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.self_attn.reset_parameters()
        self.cross_attn.reset_parameters()
        self.ffn.reset_parameters()

    def forward(
        self, 
        x: Tensor, 
        memory: Tensor,
        self_padding_mask: Tensor = None, 
        self_attn_mask: Tensor = None,
        memory_padding_mask: Tensor = None, 
        incremental_states: Dict[str, Dict[str, Tensor]] = None,
    ):
        x, *others = self.self_attn(
            x, key_padding_mask=self_padding_mask, attn_mask=self_attn_mask,
            incremental_states=incremental_states)
        x, *others = self.cross_attn(
            x, key=memory, value=memory, key_padding_mask=memory_padding_mask,
            static_kv=True, incremental_states=incremental_states)
        x = self.ffn(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(
            self,
            n_layers,
            d_model,
            n_heads,
            d_inner=2048,
            dropout=0.1,
            attn_dropout=0.,
            normalize_before=False,
            causal=True,
            layer=None
        ):
        super().__init__()

        if layer is None:
            layer = TransformerDecoderLayer(
                d_model, n_heads, d_inner, dropout, attn_dropout, normalize_before
            )
        self.layers = _get_clones(layer, N=n_layers)

        self.norm = None
        if normalize_before:
            self.norm = nn.LayerNorm(d_model)

        self.causal = causal

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers: layer.reset_parameters()

    def _maybe_get_causal_mask(self, x, incremental_states=None):
        "Mask out subsequent positions."
        if not self.causal:
            return None
        if self._inferring and incremental_states is not None:
            return None

        size = x.shape[1]
        causal_mask = torch.triu(
            torch.ones((size, size), dtype=torch.bool, device=x.device),
            diagonal=1
        )
        return causal_mask

    def forward(
        self, 
        x: Tensor, 
        memory: Tensor,
        self_padding_mask: Tensor = None, 
        memory_padding_mask: Tensor = None, 
        incremental_states: Dict[str, Dict[str, Tensor]] = None,
    ):
        out = x
        self_attn_mask = self._maybe_get_causal_mask(x, incremental_states)

        for layer in self.layers:
            out = layer(
                out, memory,
                self_padding_mask=self_padding_mask,
                self_attn_mask=self_attn_mask,
                memory_padding_mask=memory_padding_mask,
                incremental_states=incremental_states
            )

        if self.norm is not None:
            out = self.norm(out)
        return out


if __name__ == '__main__':
    from byprot.models.sequence.transformer_decoder import *

    B, L, D = 10, 7, 32
    M = 5
    decoder = TransformerDecoder(n_layers=6, d_model=D, n_heads=4, d_inner=2*D, normalize_before=True)

    x = torch.randn(B, L, D)
    mem = torch.randn(B, M, D)

    key_padding_mask = ~(torch.arange(L)[None] < torch.randint(1, L, (B,))[:, None])

    # attn_mask = torch.triu(torch.ones(10, 7, 7, dtype=torch.bool), 1)
    x = decoder(x, mem, self_padding_mask=key_padding_mask)
