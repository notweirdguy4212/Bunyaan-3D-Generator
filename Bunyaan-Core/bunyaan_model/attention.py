"""Bunyaan-native conditioning attention blocks.

Cross-attention is applied only at low-resolution feature stages so its memory
cost scales with bottleneck tokens rather than the full 3D/2D input resolution.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


def _heads(channels: int, preferred: int = 8) -> int:
    for h in range(min(preferred, channels), 0, -1):
        if channels % h == 0:
            return h
    return 1


class SpatialConditionAttention(nn.Module):
    """Let spatial features attend to native text-token features.

    Accepts [B,C,H,W] or [B,C,D,H,W] feature maps and [B,L,context_dim]
    context. Padding positions are masked. A small residual gate stabilizes
    early training while allowing the attention path to become useful.
    """
    def __init__(self, channels: int, context_dim: int, heads: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.context = nn.Linear(context_dim, channels)
        self.context_norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, _heads(channels, heads), batch_first=True)
        self.out = nn.Linear(channels, channels)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, context: torch.Tensor, padding_mask=None):
        spatial = x.shape[2:]
        original = x
        # Bound attention memory for large UV maps. The residual is computed on
        # a pooled feature grid and resized back, while the U-Net keeps full detail.
        if x.ndim == 4 and x.shape[-2] * x.shape[-1] > 4096:
            h, w = x.shape[-2:]
            x = F.adaptive_avg_pool2d(x, (min(h, 64), min(w, 64)))
        elif x.ndim == 5 and x.shape[-3] * x.shape[-2] * x.shape[-1] > 4096:
            d, h, w = x.shape[-3:]
            x = F.adaptive_avg_pool3d(x, (min(d, 16), min(h, 16), min(w, 16)))
        spatial = x.shape[2:]
        q = x.flatten(2).transpose(1, 2)
        qn = self.norm(q)
        kv = self.context_norm(self.context(context))
        attended, _ = self.attn(qn, kv, kv, key_padding_mask=padding_mask, need_weights=False)
        y = q + self.gate.tanh() * self.out(attended)
        y = y.transpose(1, 2).reshape(x.shape[0], x.shape[1], *spatial)
        delta = y - x
        if delta.shape[2:] != original.shape[2:]:
            mode = "bilinear" if delta.ndim == 4 else "trilinear"
            delta = F.interpolate(delta, size=original.shape[2:], mode=mode, align_corners=False)
        return original + delta

class LinearSpatialAttention(nn.Module):
    """Memory-efficient global self-attention over 2D or 3D feature grids.

    Factorizes attention into channel-space context and avoids constructing an
    N-by-N spatial attention matrix. This is especially useful for volumetric
    bottlenecks where quadratic attention would be prohibitively expensive.
    """
    def __init__(self, channels: int, heads: int = 8):
        super().__init__()
        heads = _heads(channels, heads)
        self.heads = heads
        self.head_dim = channels // heads
        self.norm = nn.GroupNorm(_heads(channels, 8), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor):
        shape = x.shape
        residual = x
        qkv = self.qkv(self.norm(x).flatten(2))
        b, _, n = qkv.shape
        q, k, v = qkv.reshape(b, 3, self.heads, self.head_dim, n).unbind(1)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        context = torch.matmul(k, v.transpose(-2, -1))
        attended = torch.matmul(context, q).reshape(b, shape[1], n)
        attended = self.proj(attended).reshape(shape)
        return residual + self.gate.tanh() * attended
