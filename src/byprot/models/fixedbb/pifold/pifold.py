from dataclasses import dataclass

import torch
from byprot.models import register_model
from byprot.models.fixedbb import FixedBackboneDesignEncoderDecoder
from byprot.datamodules.datasets.data_utils import Alphabet
from byprot.models.fixedbb.generator import new_arange, sample_from_categorical
from torch.nn import functional as F

from .model import PiFoldModel


@dataclass
class PiFoldConfig:
    display_step: int = 10
    d_model: int = 128
    hidden_dim: int = 128
    node_features: int = 128
    edge_features: int = 128
    k_neighbors: int = 30
    dropout: float = 0.1
    num_encoder_layers: int = 10
    updating_edges: int = 4
    node_dist: int = 1
    node_angle: int = 1
    node_direct: int = 1
    edge_dist: int = 1
    edge_angle: int = 1
    edge_direct: int = 1
    virtual_num: int = 3

    n_vocab: int = 22 
    use_esm_alphabet: bool = False

@register_model('pifold')
class PiFold(FixedBackboneDesignEncoderDecoder):
    _default_cfg = PiFoldConfig()

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        if self.cfg.use_esm_alphabet:
            alphabet = Alphabet('esm')
            self.padding_idx = alphabet.padding_idx
            self.mask_idx = alphabet.mask_idx
            self.cfg.n_vocab = len(alphabet)
        else:
            alphabet = None
            self.padding_idx = 0
            self.mask_idx = 1

        self.model = PiFoldModel(args=self.cfg)

    def forward(self, batch, return_feats=False, **kwargs):
        logits, feats = self.model(
            X=batch['coords'], 
            mask=batch['coord_mask'].float(), 
            S=batch['prev_tokens'], 
            lengths=batch.get('lengths', None))

        if return_feats:
            return logits, feats
        return logits

    def forward_encoder(self, batch):
        encoder_out = self.model.encode(
            X=batch['coords'],
            mask=batch['coord_mask'].float(),
            lengths=batch['lengths']
        )
        encoder_out['coord_mask'] = batch['coord_mask'].float()

        return encoder_out

    def forward_decoder(self, prev_decoder_out, encoder_out):
        output_tokens = prev_decoder_out['output_tokens']
        output_scores = prev_decoder_out['output_scores']
        step, max_step = prev_decoder_out['step'], prev_decoder_out['max_step']
        temperature = prev_decoder_out['temperature']
        history = prev_decoder_out['history']

        output_masks = output_tokens.eq(self.mask_idx)  # & coord_mask

        logits, _ = self.model.decode(
            prev_tokens=output_tokens,
            encoder_out=encoder_out,
        )
        _tokens, _scores = sample_from_categorical(logits, temperature=temperature)

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        history.append(output_tokens.clone())

        return dict(
            output_tokens=output_tokens,
            output_scores=output_scores,
            step=step + 1,
            max_step=max_step,
            history=history
        )

    def initialize_output_tokens(self, batch, encoder_out):
        # mask = encoder_out.get('coord_mask', None)

        prev_tokens = batch['prev_tokens']
        lengths = prev_tokens.ne(self.padding_idx).sum(1)

        initial_output_tokens = torch.full_like(prev_tokens, self.padding_idx)
        initial_output_tokens.masked_fill_(new_arange(prev_tokens) < lengths[:, None], self.mask_idx)

        # if mask is not None:
        #     initial_output_tokens = torch.where(
        #         ~mask, prev_tokens, initial_output_tokens
        #     )
        # initial_output_tokens = prev_tokens.clone()

        initial_output_scores = torch.zeros(
            *initial_output_tokens.size(), device=initial_output_tokens.device
        )

        return initial_output_tokens, initial_output_scores
