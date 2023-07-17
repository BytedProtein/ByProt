from typing import Callable
import torch
from torch import nn
from torch.nn import functional as F

def get_activation_fn(activation: str) -> Callable:
    """Returns the activation function corresponding to `activation`"""

    if activation == "relu":
        return F.relu
    elif activation == "glu":
        return F.glu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "swish":
        return torch.nn.SiLU
    else:
        raise RuntimeError("activation {} not supported".format(activation))


class FFN(nn.Module):
    """ Feed-forward neural network """

    def __init__(self,
                 d_model,
                 d_inner=None,
                 activation="gelu",
                 dropout=0.0):
        super().__init__()
        d_inner = d_inner or d_model

        self.fc1 = nn.Linear(d_model, d_inner)
        self.fc2 = nn.Linear(d_inner, d_model)
        self.activation = get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()
    
    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, x):
        """
        Args:
            x: feature to perform ffn
                :math:`(*, D)`, where D is feature dimension

        Returns:
            - feed forward output
                :math:`(*, D)`, where D is feature dimension
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x