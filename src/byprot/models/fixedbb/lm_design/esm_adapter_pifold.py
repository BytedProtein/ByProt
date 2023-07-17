from dataclasses import dataclass, field
from typing import List

import torch
from byprot.models import register_model
from byprot.models.fixedbb import FixedBackboneDesignEncoderDecoder
from byprot.models.fixedbb.generator import new_arange, sample_from_categorical
from byprot.models.fixedbb.pifold import PiFold, PiFoldConfig
from torch import nn
from torch.nn import functional as F

from .modules.esm_adapter import ProteinBertModelWithStructuralAdatper


@dataclass
class ESMAdapterPiFoldConfig:
    encoder: PiFoldConfig = field(default=PiFoldConfig())
    adapter_layer_indices: List = field(default_factory=lambda: [32, ])
    separate_loss: bool = False


@register_model('esm_adapter_pifold')
class ESMAdapterPiFold(FixedBackboneDesignEncoderDecoder):
    # _default_cfg = ESMAdapterPiFoldConfig()

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.encoder = PiFold(self.cfg.encoder)
        self.decoder = ProteinBertModelWithStructuralAdatper.from_pretrained(args=self.cfg)

        self.padding_idx = self.decoder.padding_idx
        self.mask_idx = self.decoder.mask_idx
        # self.cls_idx = self.decoder.cls_idx
        # self.eos_idx = self.decoder.eos_idx

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
            return logits
        else:
            return esm_logits, encoder_logits

    def forward_encoder(self, batch):
        encoder_logits, encoder_out = self.encoder(
            batch['coords'], batch['coord_mask'], batch['prev_tokens'],
            return_feats=True
        )

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
        history = prev_decoder_out['history']

        # output_masks = output_tokens.eq(self.mask_idx)  # & coord_mask
        output_masks = output_tokens.ne(self.padding_idx)  # & coord_mask

        esm_logits = self.decoder(
            # tokens=encoder_out['init_pred'],
            tokens=output_tokens,
            encoder_out=encoder_out,
        )['logits']

        if not getattr(self.cfg, 'separate_loss', False):
            logits = esm_logits + encoder_out['logits']
        else:
            logits = esm_logits  # + encoder_out['logits']

        _tokens, _scores = sample_from_categorical(logits, temperature=None)
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
        mask = encoder_out.get('coord_mask', None)

        prev_tokens = batch['prev_tokens']
        lengths = prev_tokens.ne(self.padding_idx).sum(1)

        initial_output_tokens = torch.full_like(prev_tokens, self.padding_idx)
        initial_output_tokens.masked_fill_(new_arange(prev_tokens) < lengths[:, None], self.mask_idx)
        # initial_output_tokens[:, 0] = self.cls_idx
        # initial_output_tokens.scatter_(1, lengths[:, None] - 1, self.eos_idx)

        initial_output_tokens = encoder_out['init_pred'].clone()
        initial_output_scores = torch.zeros(
            *initial_output_tokens.size(), device=initial_output_tokens.device
        )

        return initial_output_tokens, initial_output_scores
