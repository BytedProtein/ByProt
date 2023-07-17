"""" Multihead Attention """

import abc
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.nn import Parameter


class FairseqIncrementalState(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _incremental_state_id(self):
        # self._incremental_state_id = str(uuid.uuid4())
        return id(self)

    def _get_full_incremental_state_key(self, key: str) -> str:
        return "{}.{}".format(self._incremental_state_id, key)

    def get_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
    ) -> Optional[Dict[str, Optional[Tensor]]]:
        """Helper for getting incremental state for an nn.Module."""
        full_key = self._get_full_incremental_state_key(key)
        if incremental_state is None or full_key not in incremental_state:
            return None
        return incremental_state[full_key]

    def set_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
        value: Dict[str, Optional[Tensor]],
    ) -> Optional[Dict[str, Dict[str, Optional[Tensor]]]]:
        """Helper for setting incremental state for an nn.Module."""
        if incremental_state is not None:
            full_key = self._get_full_incremental_state_key(key)
            incremental_state[full_key] = value
        return incremental_state


def with_incremental_state(cls):
    cls.__bases__ = (FairseqIncrementalState,) + tuple(
        b for b in cls.__bases__ if b != FairseqIncrementalState
    )
    return cls


class Attention(nn.Module):
    @abc.abstractmethod
    def forward(
        self, query, key, value,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        """
            query: [bsz, qlen, d]
            key/value: [bsz, klen, d]
            key_padding_mask: [bsz, klen] (pad is True)
            return: [bsz, qlen, d]
        """
        pass


@with_incremental_state
class MHA(Attention):
    """Multi-headed attention. (from fairseq)

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        need_head_weights: bool = False,
        static_kv=False,
        incremental_states: Dict[str, Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        if incremental_states is not None:
            saved_states = self._get_input_buffer(incremental_states)
            if (
                saved_states is not None and "prev_key" in saved_states
                and static_kv
            ):
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                key, value = None, None
        else:
            saved_states = None

        q = self.q_proj(query) * self.scaling
        q = rearrange(q, 'b l (h d) -> h b l d', h=self.num_heads)
        if key is not None:
            k = self.k_proj(key)
            k = rearrange(k, 'b l (h d) -> h b l d', h=self.num_heads)
        if value is not None:
            v = self.v_proj(value)
            v = rearrange(v, 'b l (h d) -> h b l d', h=self.num_heads)

        if saved_states is not None:
            if 'prev_key' in saved_states:
                _prev_key = saved_states['prev_key']
                if static_kv:
                    k = _prev_key
                else:
                    k = torch.cat([_prev_key, k], dim=2)
            if 'prev_value' in saved_states:
                _prev_value = saved_states['prev_value']
                if static_kv:
                    v = _prev_value
                else:
                    v = torch.cat([_prev_value, v], dim=2)

            saved_states['prev_key'] = k
            saved_states['prev_value'] = v
            self._set_input_buffer(incremental_states, saved_states)

        mask = False
        if attn_mask is not None:
            # attn_mask: [qlen, klen] where True as disabling
            mask = mask | attn_mask[None, None].to(torch.bool)

        if key_padding_mask is not None:
            # don't attend to padding symbols, True as disabling
            # key_padding_mask: [bsz, klen]
            mask = mask | key_padding_mask[None, :, None, :].to(torch.bool)

        # [h, b, qlen, klen]
        logits = torch.einsum('hbqd,hbkd->hbqk', [q, k])
        if mask is not False:
            logits = logits.masked_fill(mask, -torch.inf)

        scores = F.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
        scores = self.dropout(scores)

        out = torch.einsum('hbqk,hbkd->hbqd', [scores, v])
        out = rearrange(out, 'h b l d -> b l (h d)')
        out = self.out_proj(out)

        if not need_weights:
            scores: Optional[Tensor] = None
        return out, scores

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)


if __name__ == '__main__':

    attn = MHA(32, 4, dropout=0.1, add_bias_kv=True)
    attn.eval()

    query = torch.randn(10, 7, 32)
    key = torch.randn(10, 7, 32)
    value = torch.randn(10, 7, 32)

    key_padding_mask = ~(torch.arange(7)[None] < torch.randint(1, 7, (10,))[:, None])

    attn_mask = torch.triu(torch.ones(10, 7, 7, dtype=torch.bool), 1)
    attn(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[1][0, 0]
