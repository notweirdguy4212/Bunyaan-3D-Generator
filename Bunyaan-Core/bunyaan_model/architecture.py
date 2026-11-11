"""Bunyaan native 3D latent diffusion model.

Original Bunyaan implementation. This defines a compact research architecture,
not pretrained weights. Shapes are represented as continuous occupancy volumes.
"""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F
from .attention import SpatialConditionAttention, LinearSpatialAttention


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / max(half - 1, 1))
    phase = t.float().unsqueeze(1) * freq.unsqueeze(0)
    emb = torch.cat([phase.sin(), phase.cos()], dim=1)
    return F.pad(emb, (0, dim - emb.shape[1]))


class TextConditioner(nn.Module):
    """Byte-level text encoder; avoids external pretrained text models/tokenizers."""
    def __init__(self, width=256, max_length=128, layers=4, heads=8):
        super().__init__()
        self.max_length = max_length
        self.token = nn.Embedding(257, width, padding_idx=0)
        self.position = nn.Parameter(torch.randn(1, max_length, width) * 0.01)
        block = nn.TransformerEncoderLayer(width, heads, width * 4, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)

    @staticmethod
    def tokenize(texts, max_length=128, device=None):
        rows = []
        for text in texts:
            raw = str(text).encode("utf-8")[:max_length - 2]
            rows.append([1] + [b + 2 for b in raw] + [2])
        out = torch.zeros(len(rows), max_length, dtype=torch.long, device=device)
        for i, row in enumerate(rows): out[i, :len(row)] = torch.tensor(row, device=device)
        return out

    def encode_tokens(self, tokens):
        mask = tokens.eq(0)
        x = self.token(tokens) + self.position[:, :tokens.shape[1]]
        x = self.encoder(x, src_key_padding_mask=mask)
        return self.norm(x), mask

    def forward(self, tokens):
        x, mask = self.encode_tokens(tokens)
        valid = (~mask).float().unsqueeze(-1)
        return (x * valid).sum(1) / valid.sum(1).clamp_min(1.0)


class ImageConditioner(nn.Module):
    """Small native image encoder for normalized RGB images [B,3,H,W]."""
    def __init__(self, width=256):
        super().__init__()
        channels = [3, 32, 64, 128, 192, width]
        layers = []
        for a, b in zip(channels[:-1], channels[1:]):
            layers += [nn.Conv2d(a, b, 3, stride=2, padding=1), nn.GroupNorm(8, b), nn.SiLU()]
        self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.LayerNorm(width))
    def forward(self, image): return self.net(image)


class ResBlock3D(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()
        groups = min(8, channels)
        while channels % groups: groups -= 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.film = nn.Linear(cond_dim, channels * 2)
        self.act = nn.SiLU()
    def forward(self, x, cond):
        h = self.conv1(self.act(self.norm1(x)))
        scale, shift = self.film(cond).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        return x + self.conv2(self.act(h)) * 0.2


class DownStage(nn.Module):
    def __init__(self, cin, cout, cond):
        super().__init__(); self.proj = nn.Conv3d(cin, cout, 3, padding=1)
        self.block = ResBlock3D(cout, cond); self.down = nn.Conv3d(cout, cout, 4, stride=2, padding=1)
    def forward(self, x, c):
        x = self.block(self.proj(x), c); skip = x
        return self.down(x), skip


class UpStage(nn.Module):
    def __init__(self, cin, skip, cout, cond):
        super().__init__(); self.up = nn.ConvTranspose3d(cin, cout, 4, stride=2, padding=1)
        self.mix = nn.Conv3d(cout + skip, cout, 1); self.block = ResBlock3D(cout, cond)
    def forward(self, x, skip, c):
        x = self.up(x)
        if x.shape[-3:] != skip.shape[-3:]: x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        return self.block(self.mix(torch.cat([x, skip], 1)), c)


class Bunyaan3D(nn.Module):
    """Conditional 3D U-Net denoiser over occupancy volumes.

    Input noisy_volume: [B,1,R,R,R], timestep: [B], tokens: [B,L].
    Optional image: [B,3,H,W]. Output is predicted diffusion noise [B,1,R,R,R].
    """
    def __init__(self, resolution=32, width=64, cond_dim=256, text_layers=4):
        super().__init__()
        if resolution < 16 or resolution % 8: raise ValueError("resolution must be >=16 and divisible by 8")
        self.resolution, self.width = resolution, width
        self.text = TextConditioner(cond_dim, layers=text_layers)
        self.image = ImageConditioner(cond_dim)
        self.image_gate = nn.Parameter(torch.tensor(0.0))
        self.time = nn.Sequential(nn.Linear(cond_dim, cond_dim * 4), nn.SiLU(), nn.Linear(cond_dim * 4, cond_dim))
        self.input = nn.Conv3d(1, width, 3, padding=1)
        self.d1 = DownStage(width, width, cond_dim)
        self.d2 = DownStage(width, width * 2, cond_dim)
        self.d3 = DownStage(width * 2, width * 4, cond_dim)
        self.mid1 = ResBlock3D(width * 4, cond_dim); self.mid2 = ResBlock3D(width * 4, cond_dim)
        self.text_attention = SpatialConditionAttention(width * 4, cond_dim, heads=8)
        self.global_attention = LinearSpatialAttention(width * 4, heads=8)
        self.u3 = UpStage(width * 4, width * 4, width * 2, cond_dim)
        self.u2 = UpStage(width * 2, width * 2, width, cond_dim)
        self.u1 = UpStage(width, width, width, cond_dim)
        self.out = nn.Sequential(nn.GroupNorm(8 if width % 8 == 0 else 1, width), nn.SiLU(), nn.Conv3d(width, 1, 3, padding=1))

    def forward(self, noisy_volume, timestep, tokens, image=None):
        text_context, text_padding = self.text.encode_tokens(tokens)
        cond = self.text(tokens)
        if image is not None: cond = cond + torch.sigmoid(self.image_gate) * self.image(image)
        cond = cond + self.time(timestep_embedding(timestep, cond.shape[-1]))
        x = self.input(noisy_volume)
        x, s1 = self.d1(x, cond); x, s2 = self.d2(x, cond); x, s3 = self.d3(x, cond)
        x = self.mid2(self.mid1(x, cond), cond)
        x = self.text_attention(x, text_context, text_padding)
        x = self.global_attention(x)
        x = self.u3(x, s3, cond); x = self.u2(x, s2, cond); x = self.u1(x, s1, cond)
        return self.out(x)


class BunyaanDiffusion:
    def __init__(self, steps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.steps = int(steps)
        self.betas = torch.linspace(beta_start, beta_end, self.steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, 0)
    def add_noise(self, clean, noise, t):
        a = self.alpha_bars[t].view(-1, 1, 1, 1, 1)
        return a.sqrt() * clean + (1 - a).sqrt() * noise
    @torch.no_grad()
    def sample(self, model, tokens, image=None, shape=None, guidance=1.0, device=None):
        device = device or self.betas.device
        b = tokens.shape[0]; r = shape or model.resolution
        x = torch.randn(b, 1, r, r, r, device=device)
        for step in reversed(range(self.steps)):
            t = torch.full((b,), step, device=device, dtype=torch.long)
            eps = model(x, t, tokens, image)
            if guidance != 1.0:
                null_tokens = torch.zeros_like(tokens)
                null_tokens[:, 0] = 1  # BOS-only unconditional prompt, also used during training dropout.
                uncond = model(x, t, null_tokens, image)
                eps = uncond + guidance * (eps - uncond)
            alpha, beta = self.alphas[step], self.betas[step]
            abar = self.alpha_bars[step]
            mean = (x - beta / torch.sqrt(1 - abar) * eps) / torch.sqrt(alpha)
            if step: x = mean + torch.sqrt(beta) * torch.randn_like(x)
            else: x = mean
        return x.sigmoid()


    @torch.no_grad()
    def sample_ddim(self, model, tokens, image=None, shape=None, steps=50, guidance=4.0,
                    eta=0.0, device=None, generator=None):
        """Deterministic/low-stochasticity DDIM sampler for faster inference."""
        if steps < 1 or steps > self.steps: raise ValueError("steps must be in [1, diffusion steps]")
        if eta < 0: raise ValueError("eta must be non-negative")
        device = torch.device(device or self.betas.device)
        model = model.to(device).eval(); tokens = tokens.to(device)
        image = image.to(device) if image is not None else None
        batch = tokens.shape[0]; resolution = shape or model.resolution
        x = torch.randn((batch,1,resolution,resolution,resolution), device=device, generator=generator)
        indices = torch.linspace(self.steps-1, 0, steps, device=device).long().unique_consecutive()
        for i, step in enumerate(indices):
            t = torch.full((batch,), int(step), device=device, dtype=torch.long)
            eps = model(x, t, tokens, image)
            if guidance != 1.0:
                null = torch.zeros_like(tokens); null[:,0] = 1
                eps0 = model(x, t, null, image)
                eps = eps0 + guidance * (eps - eps0)
            abar = self.alpha_bars[step]
            x0 = (x - (1-abar).sqrt()*eps) / abar.sqrt().clamp_min(1e-8)
            if i + 1 == len(indices):
                x = x0
                continue
            prev = self.alpha_bars[indices[i+1]]
            sigma = eta * torch.sqrt(((1-prev)/(1-abar) * (1-abar/prev)).clamp_min(0))
            direction = torch.sqrt((1-prev-sigma.square()).clamp_min(0))*eps
            noise = torch.randn(x.shape, device=device, dtype=x.dtype, generator=generator) if eta else 0.0
            x = prev.sqrt()*x0 + direction + sigma*noise
        return x.sigmoid()


def save_checkpoint(path, model, optimizer=None, epoch=0, config=None, ema_model=None, extra=None):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": "bunyaan-native-3d", "version": "1.4.0-dev", "epoch": epoch,
               "config": config or {}, "model": model.state_dict()}
    if optimizer is not None: payload["optimizer"] = optimizer.state_dict()
    if ema_model is not None: payload["ema"] = ema_model.state_dict()
    if extra: payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, device="cpu", use_ema=True):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != "bunyaan-native-3d": raise ValueError("Not a Bunyaan native checkpoint")
    cfg = payload.get("config", {})
    model = Bunyaan3D(**{k: cfg[k] for k in ("resolution", "width", "cond_dim", "text_layers") if k in cfg})
    state = payload.get("ema") if use_ema else None
    model.load_state_dict(state or payload["model"], strict=False); model.to(device).eval()
    return model, payload
