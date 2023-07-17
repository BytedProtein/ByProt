import itertools
import math
import os
import pickle
import re
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple, Union, Mapping

import torch
from torch import nn


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


class IterativeRefinementGenerator(object):
    def __init__(self,
                 alphabet=None,
                 max_iter=1,
                 strategy='denoise',
                 temperature=None,
                 **kwargs
                 ):

        self.alphabet = alphabet
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx

        self.max_iter = max_iter
        self.strategy = strategy
        self.temperature = temperature

    @torch.no_grad()
    def generate(self, model, batch, alphabet=None, 
                 max_iter=None, strategy=None, temperature=None, replace_visible_tokens=False, 
                 need_attn_weights=False):
        alphabet = alphabet or self.alphabet
        max_iter = max_iter or self.max_iter
        strategy = strategy or self.strategy
        temperature = temperature or self.temperature

        # 0) encoding
        encoder_out = model.forward_encoder(batch)

        # 1) initialized from all mask tokens
        initial_output_tokens, initial_output_scores = model.initialize_output_tokens(
            batch, encoder_out=encoder_out)
        prev_decoder_out = dict(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            output_masks=None,
            attentions=None,
            step=0,
            max_step=max_iter,
            history=[initial_output_tokens.clone()],
            temperature=temperature,
        )

        if need_attn_weights:
            attns = [] # list of {'in', 'out', 'attn'} for all iteration

        if strategy == 'discrete_diffusion':
            prev_decoder_out['output_masks'] = model.get_non_special_sym_mask(batch['prev_tokens'])

        # iterative refinement
        for step in range(max_iter):

            # 2.1: predict
            decoder_out = model.forward_decoder(
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

        # skeptical_mask = _skeptical_unmasking(
        #     output_scores=output_scores,
        #     output_masks=output_tokens.ne(self.padding_idx),  # & coord_mask,
        #     p=0.08
        # )

        # output_tokens.masked_fill_(skeptical_mask, self.alphabet.unk_idx)
        # output_scores.masked_fill_(skeptical_mask, 0.0)
        decoder_out = prev_decoder_out

        if need_attn_weights:
            return decoder_out['output_tokens'], decoder_out['output_scores'], attns
        return decoder_out['output_tokens'], decoder_out['output_scores']


def sample_from_categorical(logits=None, temperature=1.0):
    if temperature:
        dist = torch.distributions.Categorical(logits=logits.div(temperature))
        tokens = dist.sample()
        scores = dist.log_prob(tokens)
    else:
        scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores
