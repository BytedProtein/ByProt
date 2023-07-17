import math

import torch
from torch import nn


class Embedding(nn.Embedding):
    def __init__(self, vocab_size, d_model, padding_idx=None):
        super().__init__(vocab_size, d_model, padding_idx=padding_idx)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx

        nn.init.normal_(self.weight, mean=0, std=self.d_model ** -0.5)
        nn.init.constant_(self.weight[padding_idx], 0)


class PositionEmbedding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout=0, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.scaling = self.d_model ** 0.5

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def reset_parameters(self):
        pass

    def get_as(self, x):
        # x: [bsz, len, d_model]
        return self.pe[:, :x.size(1)].requires_grad_(False)

    def forward(self, x):
        x = x * self.scaling + self.get_as(x)
        return self.dropout(x)


class LearnedPositionEmbedding(nn.Module):
    def __init__(self, d_model, dropout, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)

        self.reset_parameters()

    def reset_parameters(self):
        self.pe.reset_parameters()

    def get_as(self, x):
        return self.pe(x)

    def forward(self, x):
        x = x + self.get_as(x)
        return self.dropout(x)


registry = {
    'default': PositionEmbedding,
    'learned': LearnedPositionEmbedding
}
