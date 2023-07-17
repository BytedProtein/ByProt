import torch
from torch import nn
import esm

class GVPTransformerEncoderWrapper(nn.Module):
    def __init__(self, alphabet, freeze=True):
        super().__init__()
        _model, _alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        self.encoder = _model.encoder
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad_(False)

        self.embed_dim = self.encoder.embed_tokens.embedding_dim
        self.out_proj = nn.Linear(self.embed_dim, len(alphabet))

    def forward(self, batch, **kwargs):
        return_all_hiddens = False
        padding_mask = torch.isnan(batch['coords'][:, :, 0, 0])
        coords = batch['coords'][:, :, :3, :]
        confidence = torch.ones(batch['coords'].shape[0:2]).to(coords.device)
        encoder_out = self.encoder(coords, padding_mask, confidence,
            return_all_hiddens=return_all_hiddens)
        # encoder_out['encoder_out'][0] = torch.transpose(encoder_out['encoder_out'][0], 0, 1)
        encoder_out['feats'] = encoder_out['encoder_out'][0].transpose(0, 1)
        logits = self.out_proj(encoder_out['feats'])
        return logits, encoder_out
    
