"""Generate a mesh and vertex PBR attributes from a trained Bunyaan v1.5 checkpoint."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from bunyaan_model.implicit3d import load_implicit_checkpoint
from bunyaan_model.implicit_mesh import implicit_field_to_mesh


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True); p.add_argument("--input", required=True)
    p.add_argument("--out", required=True, help=".npz or .glb; GLB also writes a PBR attribute sidecar .npz")
    p.add_argument("--resolution", type=int, default=128); p.add_argument("--chunk-size", type=int, default=65536)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(); device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    model, text, views, _ = load_implicit_checkpoint(a.checkpoint, device=device, use_ema=True)
    item = torch.load(a.input, map_location="cpu", weights_only=False)
    if "coarse_volume" not in item: raise ValueError("Input .pt must contain coarse_volume")
    result = implicit_field_to_mesh(model, text, item["coarse_volume"], item.get("caption", ""),
        item.get("images"), item.get("cameras"), views, a.resolution, a.chunk_size)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".npz":
        np.savez_compressed(out, **result)
    elif out.suffix.lower() == ".glb":
        try: import trimesh
        except ImportError as exc: raise RuntimeError("Install trimesh to export GLB") from exc
        colors = np.concatenate([np.clip(result["base_color"],0,1), np.clip(result["opacity"],0,1)], axis=1)
        vertex_colors = np.round(colors * 255).astype(np.uint8)
        mesh = trimesh.Trimesh(vertices=result["vertices"], faces=result["faces"], process=False)
        mesh.visual.vertex_colors = vertex_colors
        # GLB stores base colour/opacity as vertex colours. Per-vertex roughness,
        # metallic and the complete field data are preserved in the sidecar NPZ.
        mesh.export(out)
        np.savez_compressed(out.with_suffix(".pbr.npz"), **result)
    else: raise ValueError("--out must end in .npz or .glb")
    print(f"Saved {out} ({len(result['vertices'])} vertices, {len(result['faces'])} faces)")

if __name__ == "__main__": main()
