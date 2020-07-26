import torch
import torch.nn as nn
import torch.nn.functional as F


class Quantizer(nn.Module):
    def __init__(self, e_dim, n_embed):
        super().__init__()
        embed = torch.randn(e_dim, n_embed)
        self.register_buffer("embed", embed) # don't optimize by training

    def forward(self, x):
        e_dim, n_embed = self.embed.shape
        # 1. get encoded tensor x: (B, C, H, W)
        # 2. flatten x into (B*H*W,C)
        x = x.permute(0, 2, 3, 1)
        flatten = x.reshape(-1, e_dim)

        # distance = (f - e)^2 = f^2- 2*f*e + e^2
        distance = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_idx = (-distance).max(1)
        embed_onehot = F.one_hot(embed_idx, n_embed).to(flatten.dtype)
        embed_idx = embed_idx.view(*x.shape[:-1])
        quantize = self.embed_code(embed_idx)
        return quantize

    def embed_code(self, embed_idx):
        return F.embedding(embed_idx, self.embed.transpose(0, 1))
