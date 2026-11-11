"""Bunyaan v1.5 implicit 3D field and native multi-view conditioning.

Independent research architecture; no TRELLIS/Hunyuan runtime or source imports.
"""
from __future__ import annotations
import math
from typing import Dict, Optional
import torch
from torch import nn
import torch.nn.functional as F


class FourierPositionEncoding(nn.Module):
    """Multi-frequency encoding for normalized 3D query coordinates."""
    def __init__(self, bands: int = 8):
        super().__init__()
        if bands < 1: raise ValueError("bands must be positive")
        self.register_buffer("frequencies", (2.0 ** torch.arange(bands)) * math.pi)

    @property
    def output_dim(self): return 3 * (2 * len(self.frequencies) + 1)

    def forward(self, points):
        if points.shape[-1] != 3: raise ValueError("query points must end in xyz")
        phase = points.unsqueeze(-1) * self.frequencies
        return torch.cat([points, phase.sin().flatten(-2), phase.cos().flatten(-2)], -1)


class BunyaanCoarseFeatureEncoder(nn.Module):
    """Encode a coarse generated volume into a compact 3D feature grid."""
    def __init__(self, in_channels=1, width=32, feature_dim=128):
        super().__init__()
        def groups(ch):
            g = min(8, ch)
            while ch % g: g -= 1
            return g
        self.stem = nn.Sequential(nn.Conv3d(in_channels, width, 3, padding=1),
                                  nn.GroupNorm(groups(width), width), nn.SiLU())
        self.res1 = nn.Sequential(nn.Conv3d(width, width, 3, padding=1),
                                  nn.GroupNorm(groups(width), width), nn.SiLU(),
                                  nn.Conv3d(width, width, 3, padding=1))
        self.down1 = nn.Sequential(nn.Conv3d(width, width*2, 4, stride=2, padding=1),
                                   nn.GroupNorm(groups(width*2), width*2), nn.SiLU())
        self.res2 = nn.Sequential(nn.Conv3d(width*2, width*2, 3, padding=1),
                                  nn.GroupNorm(groups(width*2), width*2), nn.SiLU(),
                                  nn.Conv3d(width*2, width*2, 3, padding=1))
        self.down2 = nn.Sequential(nn.Conv3d(width*2, feature_dim, 4, stride=2, padding=1),
                                   nn.GroupNorm(groups(feature_dim), feature_dim), nn.SiLU())

    def forward(self, coarse_volume):
        if coarse_volume.ndim != 5 or min(coarse_volume.shape[-3:]) < 16:
            raise ValueError("coarse_volume must be [B,C,D,H,W] with spatial dimensions >=16")
        x = self.stem(coarse_volume)
        x = x + self.res1(x) * 0.2
        x = self.down1(x)
        x = x + self.res2(x) * 0.2
        return self.down2(x)


class BunyaanHierarchicalImplicitModel(nn.Module):
    """Coarse volume -> compact feature grid -> arbitrary-resolution implicit field."""
    def __init__(self, in_channels=1, encoder_width=32, feature_dim=128,
                 hidden_dim=256, condition_dim=256):
        super().__init__()
        self.encoder = BunyaanCoarseFeatureEncoder(in_channels, encoder_width, feature_dim)
        self.decoder = BunyaanImplicitFieldDecoder(feature_dim, hidden_dim,
                                                    condition_dim=condition_dim)

    def forward(self, coarse_volume, query_points, condition_tokens=None,
                condition_padding_mask=None):
        feature_grid = self.encoder(coarse_volume)
        return self.decoder(feature_grid, query_points, condition_tokens,
                            condition_padding_mask)


class BunyaanImplicitFieldDecoder(nn.Module):
    """Query a compact 3D feature grid at arbitrary coordinates.

    Predicts SDF, normals, linear base colour, roughness, metallic and opacity.
    Query coordinates are xyz in [-1,1].
    """
    output_channels = ("sdf", "normal_x", "normal_y", "normal_z", "base_color_r",
                       "base_color_g", "base_color_b", "roughness", "metallic", "opacity")

    def __init__(self, feature_dim=128, hidden_dim=256, position_bands=8, condition_dim=None):
        super().__init__()
        if feature_dim < 8 or hidden_dim < 32 or hidden_dim % 8:
            raise ValueError("feature_dim >= 8 and hidden_dim >= 32 divisible by 8 required")
        self.feature_dim = feature_dim
        self.position = FourierPositionEncoding(position_bands)
        self.condition_projection = nn.Linear(condition_dim, hidden_dim) if condition_dim else None
        self.condition_attention = nn.MultiheadAttention(hidden_dim, 8, batch_first=True) if condition_dim else None
        self.input = nn.Sequential(nn.Linear(feature_dim * 2 + self.position.output_dim, hidden_dim),
                                   nn.LayerNorm(hidden_dim), nn.GELU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                                   nn.Linear(hidden_dim, hidden_dim)) for _ in range(3)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(3)])
        self.output = nn.Linear(hidden_dim, 10)

    def forward(self, feature_grid, query_points, condition_tokens=None, condition_padding_mask=None):
        if feature_grid.ndim != 5 or feature_grid.shape[1] != self.feature_dim:
            raise ValueError(f"feature_grid must be [B,{self.feature_dim},D,H,W]")
        if query_points.ndim != 3 or query_points.shape[0] != feature_grid.shape[0] or query_points.shape[-1] != 3:
            raise ValueError("query_points must have shape [B,N,3]")
        if torch.any(query_points < -1.0001) or torch.any(query_points > 1.0001):
            raise ValueError("query coordinates must lie in [-1,1]")
        # Explicit trilinear interpolation supports the second derivatives needed
        # by Eikonal regularization (3D grid_sample backward-backward is not
        # implemented by every PyTorch backend).
        b, channels, depth, height, width = feature_grid.shape
        xyz = (query_points.clamp(-1, 1) + 1) * 0.5
        px = xyz[..., 0] * (width - 1)
        py = xyz[..., 1] * (height - 1)
        pz = xyz[..., 2] * (depth - 1)
        x0, y0, z0 = px.floor().long(), py.floor().long(), pz.floor().long()
        x1, y1, z1 = (x0 + 1).clamp(max=width-1), (y0 + 1).clamp(max=height-1), (z0 + 1).clamp(max=depth-1)
        wx, wy, wz = px - x0, py - y0, pz - z0
        flat = feature_grid.reshape(b, channels, -1)
        def corner(x, y, z):
            idx = (z * height * width + y * width + x).reshape(b, 1, -1).expand(-1, channels, -1)
            return flat.gather(2, idx).transpose(1, 2)
        c000, c100 = corner(x0,y0,z0), corner(x1,y0,z0)
        c010, c110 = corner(x0,y1,z0), corner(x1,y1,z0)
        c001, c101 = corner(x0,y0,z1), corner(x1,y0,z1)
        c011, c111 = corner(x0,y1,z1), corner(x1,y1,z1)
        wx, wy, wz = wx.unsqueeze(-1), wy.unsqueeze(-1), wz.unsqueeze(-1)
        c00 = c000*(1-wx) + c100*wx; c10 = c010*(1-wx) + c110*wx
        c01 = c001*(1-wx) + c101*wx; c11 = c011*(1-wx) + c111*wx
        c0 = c00*(1-wy) + c10*wy; c1 = c01*(1-wy) + c11*wy
        local = c0*(1-wz) + c1*wz
        global_feature = feature_grid.mean((2, 3, 4))[:, None, :].expand_as(local)
        hidden = self.input(torch.cat([local, global_feature, self.position(query_points)], -1))
        if condition_tokens is not None:
            if self.condition_attention is None: raise ValueError("decoder must be constructed with condition_dim")
            context = self.condition_projection(condition_tokens)
            attended, _ = self.condition_attention(hidden, context, context,
                key_padding_mask=condition_padding_mask, need_weights=False)
            hidden = hidden + attended
        for block, norm in zip(self.blocks, self.norms): hidden = norm(hidden + block(hidden))
        raw = self.output(hidden)
        sdf = raw[..., 0:1]
        return {"sdf": sdf, "occupancy": torch.sigmoid(-sdf * 32.0),
                "normal": F.normalize(raw[..., 1:4], dim=-1, eps=1e-6),
                "base_color": raw[..., 4:7].sigmoid(),
                "roughness": raw[..., 7:8].sigmoid().clamp(0.04, 1.0),
                "metallic": raw[..., 8:9].sigmoid(), "opacity": raw[..., 9:10].sigmoid()}


class BunyaanMultiViewEncoder(nn.Module):
    """Native camera-aware image-set encoder. Inputs [B,V,3,H,W], cameras [B,V,3,4]."""
    def __init__(self, width=256, patch_grid=4, layers=4, heads=8):
        super().__init__()
        if width % heads: raise ValueError("width must be divisible by heads")
        self.patch_grid = patch_grid
        channels = [3, 32, 64, 128, width]; conv = []
        for a, b in zip(channels[:-1], channels[1:]):
            groups = min(8, b)
            while b % groups: groups -= 1
            conv += [nn.Conv2d(a, b, 3, stride=2, padding=1), nn.GroupNorm(groups, b), nn.GELU()]
        self.image_encoder = nn.Sequential(*conv)
        self.camera_encoder = nn.Sequential(nn.Linear(12, width), nn.GELU(), nn.Linear(width, width))
        self.patch_position = nn.Parameter(torch.randn(1, patch_grid * patch_grid, width) * 0.01)
        block = nn.TransformerEncoderLayer(width, heads, width * 4, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.fusion = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, images, cameras, view_padding_mask=None):
        if images.ndim != 5 or images.shape[2] != 3: raise ValueError("images must be [B,V,3,H,W]")
        b, v, _, h, w = images.shape
        if cameras.shape != (b, v, 3, 4): raise ValueError("cameras must be [B,V,3,4]")
        if view_padding_mask is None: view_padding_mask = torch.zeros((b,v), dtype=torch.bool, device=images.device)
        if view_padding_mask.shape != (b,v) or torch.all(view_padding_mask, dim=1).any():
            raise ValueError("view mask must be [B,V] with at least one valid view per sample")
        features = self.image_encoder(images.reshape(b*v, 3, h, w))
        features = F.adaptive_avg_pool2d(features, (self.patch_grid, self.patch_grid))
        patches = features.flatten(2).transpose(1,2).reshape(b, v*self.patch_grid**2, -1)
        patches = patches + self.patch_position.expand(b * v, -1, -1).reshape_as(patches)
        camera = self.camera_encoder(cameras.reshape(b,v,12))
        camera = camera[:,:,None,:].expand(-1,-1,self.patch_grid**2,-1).reshape_as(patches)
        padding = view_padding_mask[:,:,None].expand(-1,-1,self.patch_grid**2).reshape(b,-1)
        return self.norm(self.fusion(patches + camera, src_key_padding_mask=padding)), padding


def query_field_in_chunks(decoder, feature_grid, query_points, chunk_size=65536,
                          condition_tokens=None, condition_padding_mask=None):
    """Evaluate dense or irregular point sets without allocating all query activations."""
    if chunk_size < 1 or query_points.ndim != 3: raise ValueError("invalid chunk size or query_points")
    parts = [decoder(feature_grid, query_points[:, i:i+chunk_size], condition_tokens,
                     condition_padding_mask) for i in range(0, query_points.shape[1], chunk_size)]
    if not parts: return {}
    return {k: torch.cat([p[k] for p in parts], dim=1) for k in parts[0]}


def implicit_field_losses(prediction: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor],
                          query_points=None, weights=None):
    """SDF, normal and PBR supervision with optional Eikonal regularization."""
    weights = weights or {}
    required = ("sdf", "normal", "base_color", "roughness", "metallic", "opacity")
    missing = [k for k in required if k not in target]
    if missing: raise ValueError("Missing target fields: " + ", ".join(missing))
    losses = {"sdf": F.smooth_l1_loss(prediction["sdf"], target["sdf"]),
              "normal": (1-F.cosine_similarity(prediction["normal"],
                  F.normalize(target["normal"], dim=-1, eps=1e-6), dim=-1)).mean(),
              "base_color": F.l1_loss(prediction["base_color"], target["base_color"]),
              "roughness": F.l1_loss(prediction["roughness"], target["roughness"]),
              "metallic": F.l1_loss(prediction["metallic"], target["metallic"]),
              "opacity": F.l1_loss(prediction["opacity"], target["opacity"])}
    if query_points is not None and query_points.requires_grad:
        grad = torch.autograd.grad(prediction["sdf"].sum(), query_points, create_graph=True,
                                   retain_graph=True, only_inputs=True)[0]
        losses["eikonal"] = ((grad.norm(dim=-1)-1)**2).mean()
    losses["total"] = sum(losses[k] * float(weights.get(k,1.0)) for k in losses)
    return losses


def load_implicit_checkpoint(path, device="cpu", use_ema=True):
    """Load a Bunyaan v1.5 implicit-field checkpoint and its native conditioners."""
    from pathlib import Path
    from .architecture import TextConditioner
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("format") != "bunyaan-implicit-field":
        raise ValueError("Not a Bunyaan implicit-field checkpoint")
    cfg = payload.get("config", {})
    model = BunyaanHierarchicalImplicitModel(
        feature_dim=int(cfg.get("feature_dim", 128)),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        condition_dim=int(cfg.get("condition_dim", 256)),
    )
    model_state = payload.get("ema_field_model") if use_ema else None
    model.load_state_dict(model_state or payload["field_model"], strict=True)
    text = TextConditioner(int(cfg.get("condition_dim", 256)))
    text_state = payload.get("ema_text") if use_ema else None
    text.load_state_dict(text_state or payload["text"], strict=True)
    views = None
    if cfg.get("has_views"):
        views = BunyaanMultiViewEncoder(int(cfg.get("condition_dim", 256)))
        view_state = payload.get("ema_views") if use_ema else None
        views.load_state_dict(view_state or payload["views"], strict=True)
        views.to(device).eval()
    model.to(device).eval(); text.to(device).eval()
    return model, text, views, payload
