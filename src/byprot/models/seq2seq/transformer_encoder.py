""" Transformer encoder """

import copy
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .modules.multihead_attention import MHA
from .modules.ffn import FFN
from .modules.utils import ResNorm, _get_clones


class TransformerEncoderLayer(nn.Module):
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

        self.ffn = ResNorm(
            net=FFN(d_model=d_model, d_inner=d_inner, dropout=dropout),
            dim=d_model, normalize_before=normalize_before, dropout=dropout
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.self_attn.reset_parameters()
        self.ffn.reset_parameters()

    def forward(self, x: Tensor, self_padding_mask: Tensor = None, attn_mask: Tensor = None):
        x, *others = self.self_attn(x, key_padding_mask=self_padding_mask, attn_mask=attn_mask)
        x = self.ffn(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            n_layers,
            d_model,
            n_heads,
            d_inner=2048,
            dropout=0.1,
            attn_dropout=0.,
            normalize_before=False,
            layer=None,
        ):
        super().__init__()

        if layer is None:
            layer = TransformerEncoderLayer(
                d_model, n_heads, d_inner, dropout, attn_dropout, normalize_before
            )
        self.layers = _get_clones(layer, N=n_layers)

        self.norm = None
        if normalize_before:
            self.norm = nn.LayerNorm(d_model)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers: layer.reset_parameters()

    def forward(self, x, padding_mask):
        out = x
        for layer in self.layers:
            out = layer(out, self_padding_mask=padding_mask)

        if self.norm is not None:
            out = self.norm(out)
        return out


if __name__ == '__main__':

    B, L, D = 10, 7, 32
    encoder = TransformerEncoder(n_layers=6, d_model=D, n_heads=4, d_inner=2*D, normalize_before=True)
    
    x = torch.randn(B, L, D)

    key_padding_mask = ~(torch.arange(L)[None] < torch.randint(1, L, (B,))[:, None])

    # attn_mask = torch.triu(torch.ones(10, 7, 7, dtype=torch.bool), 1)
    x = encoder(x, padding_mask=key_padding_mask)

    print(x)