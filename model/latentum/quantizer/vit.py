import torch
import torch.nn as nn
from .vit_basic import Block, precompute_freqs_cis_2d


class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.depth = config.depth
        self.num_heads = config.num_heads
        self.grid_size = config.grid_size

        self.precompute_pos = dict()
        self.input_proj = nn.Linear(config.input_feature_dim, config.hidden_size)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.blocks = nn.ModuleList([Block(config.hidden_size, config.num_heads) for _ in range(config.depth)])
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.output_proj = nn.Linear(config.hidden_size, config.output_dim)

    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        else:
            pos = precompute_freqs_cis_2d(self.hidden_size // self.num_heads, height, width).to(device)
            self.precompute_pos[(height, width)] = pos
            return pos

    def forward(self, x):
        """
        x: [B, L, D], original vit feature
        """
        pos = self.fetch_pos(self.grid_size, self.grid_size, x.device)
        B, L, D = x.shape

        x = self.input_proj(x)
        x = self.norm1(x).to(x.dtype)
        for block in self.blocks:
            x = block(x, pos)
        x = self.norm2(x)
        x = self.output_proj(x)

        return x
    
    # def forward_ste(self, x):
    #     x_low_dim = self.forward(x)
    #     p = torch.sigmoid(x_low_dim)
    #     p_ = (p > 0.5).to(x.dtype)
    #     feature_bin = p + (p_ - p).detach() # (B, 256, 16)

    #     return feature_bin
    
    # @torch.no_grad()
    # def get_code(self, x):
    #     x_low_dim = self.forward(x)
    #     p = torch.sigmoid(x_low_dim)
    #     p_ = (p > 0.5).to(x.dtype)

    #     return p_
