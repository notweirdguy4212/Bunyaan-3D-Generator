"""Chunked mesh extraction from Bunyaan's continuous implicit SDF/PBR field."""
from __future__ import annotations
import numpy as np
import torch
from .implicit3d import query_field_in_chunks
from .architecture import TextConditioner


@torch.inference_mode()
def implicit_field_to_mesh(model, text_encoder, coarse_volume, caption="",
                           images=None, cameras=None, view_encoder=None,
                           resolution=128, chunk_size=65536):
    """Sample an implicit field, extract its zero level set, and query vertex PBR.

    Returns numpy arrays for vertices/faces/normals and per-vertex PBR channels.
    Geometry sampling is bounded by chunk_size; the scalar SDF volume is retained
    in memory for marching cubes.
    """
    if resolution < 16 or chunk_size < 1: raise ValueError("resolution >=16 and chunk_size >=1 required")
    device = next(model.parameters()).device
    coarse_volume = coarse_volume.to(device)
    if coarse_volume.ndim == 4: coarse_volume = coarse_volume.unsqueeze(0)
    if coarse_volume.shape[0] != 1: raise ValueError("mesh extraction currently expects batch size 1")
    tokens = TextConditioner.tokenize([caption], device=device)
    text_context, text_mask = text_encoder.encode_tokens(tokens)
    contexts, masks = [text_context], [text_mask]
    if images is not None:
        if view_encoder is None or cameras is None:
            raise ValueError("images require a trained view_encoder and calibrated cameras")
        if images.ndim == 4: images = images.unsqueeze(0)
        if cameras.ndim == 3: cameras = cameras.unsqueeze(0)
        view_context, view_mask = view_encoder(images.to(device), cameras.to(device))
        contexts.append(view_context); masks.append(view_mask)
    condition = torch.cat(contexts, dim=1); padding = torch.cat(masks, dim=1)
    feature_grid = model.encoder(coarse_volume)

    total = resolution ** 3
    sdf = np.empty(total, dtype=np.float32)
    for start in range(0, total, chunk_size):
        idx = torch.arange(start, min(start + chunk_size, total), device=device)
        x = idx.remainder(resolution)
        y = torch.div(idx, resolution, rounding_mode="floor").remainder(resolution)
        z = torch.div(idx, resolution * resolution, rounding_mode="floor")
        points = torch.stack([x, y, z], dim=-1).float() * (2.0 / (resolution - 1)) - 1.0
        pred = model.decoder(feature_grid, points.unsqueeze(0), condition, padding)
        sdf[start:start + len(idx)] = pred["sdf"][0, :, 0].float().cpu().numpy()
    volume = sdf.reshape(resolution, resolution, resolution)
    if volume.min() > 0 or volume.max() < 0:
        raise RuntimeError("The predicted SDF does not cross zero; improve training or adjust the field/checkpoint")
    try:
        from skimage.measure import marching_cubes
    except ImportError as exc:
        raise RuntimeError("Install scikit-image to extract implicit meshes") from exc
    verts_zyx, faces, _, _ = marching_cubes(volume, level=0.0)
    vertices = verts_zyx[:, [2, 1, 0]].astype(np.float32) * (2.0 / (resolution - 1)) - 1.0
    vertex_points = torch.from_numpy(vertices).to(device).unsqueeze(0)
    attributes = query_field_in_chunks(model.decoder, feature_grid, vertex_points,
                                       chunk_size=chunk_size, condition_tokens=condition,
                                       condition_padding_mask=padding)
    return {
        "vertices": vertices,
        "faces": faces.astype(np.int32),
        "normals": attributes["normal"][0].float().cpu().numpy(),
        "base_color": attributes["base_color"][0].float().cpu().numpy(),
        "roughness": attributes["roughness"][0].float().cpu().numpy(),
        "metallic": attributes["metallic"][0].float().cpu().numpy(),
        "opacity": attributes["opacity"][0].float().cpu().numpy(),
        "field_resolution": np.asarray([resolution], dtype=np.int32),
    }
