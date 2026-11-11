import torch
from bunyaan_model.implicit3d import (
    BunyaanImplicitFieldDecoder, BunyaanMultiViewEncoder,
    query_field_in_chunks, implicit_field_losses,
)


def test_implicit_decoder_outputs_geometry_and_pbr():
    model = BunyaanImplicitFieldDecoder(feature_dim=16, hidden_dim=64, condition_dim=32)
    grid = torch.randn(2, 16, 8, 8, 8)
    points = torch.rand(2, 37, 3) * 2 - 1
    context = torch.randn(2, 5, 32)
    padding = torch.zeros(2, 5, dtype=torch.bool)
    out = model(grid, points, context, padding)
    assert out["sdf"].shape == (2, 37, 1)
    assert out["normal"].shape == (2, 37, 3)
    assert out["base_color"].shape == (2, 37, 3)
    assert torch.isfinite(out["sdf"]).all()
    assert torch.allclose(out["normal"].norm(dim=-1), torch.ones(2, 37), atol=1e-4)
    for key in ("roughness", "metallic", "opacity", "occupancy"):
        assert out[key].min() >= 0 and out[key].max() <= 1


def test_chunked_field_query_matches_full_query():
    torch.manual_seed(4)
    model = BunyaanImplicitFieldDecoder(feature_dim=16, hidden_dim=64).eval()
    grid = torch.randn(1, 16, 6, 6, 6)
    points = torch.rand(1, 29, 3) * 2 - 1
    full = model(grid, points)
    chunked = query_field_in_chunks(model, grid, points, chunk_size=7)
    for key in full:
        assert torch.allclose(full[key], chunked[key], atol=1e-6)


def test_camera_aware_multiview_fusion_masks_padded_views():
    model = BunyaanMultiViewEncoder(width=32, patch_grid=2, layers=1, heads=4)
    images = torch.randn(2, 3, 3, 64, 64)
    cameras = torch.randn(2, 3, 3, 4)
    mask = torch.tensor([[False, False, True], [False, True, True]])
    tokens, padding = model(images, cameras, mask)
    assert tokens.shape == (2, 12, 32)
    assert padding.shape == (2, 12)
    assert padding[0, 8:].all() and padding[1, 4:].all()
    assert torch.isfinite(tokens).all()


def test_implicit_field_losses_include_eikonal_term():
    model = BunyaanImplicitFieldDecoder(feature_dim=16, hidden_dim=64)
    grid = torch.randn(1, 16, 5, 5, 5)
    points = (torch.rand(1, 12, 3) * 1.8 - .9).requires_grad_(True)
    pred = model(grid, points)
    target = {"sdf": torch.zeros_like(pred["sdf"]),
              "normal": torch.randn_like(pred["normal"]),
              "base_color": torch.rand_like(pred["base_color"]),
              "roughness": torch.rand_like(pred["roughness"]),
              "metallic": torch.rand_like(pred["metallic"]),
              "opacity": torch.rand_like(pred["opacity"])}
    losses = implicit_field_losses(pred, target, points)
    assert "eikonal" in losses and torch.isfinite(losses["total"])
    losses["total"].backward()
    assert points.grad is not None and torch.isfinite(points.grad).all()


def test_hierarchical_model_encodes_coarse_volume_and_decodes_field():
    from bunyaan_model.implicit3d import BunyaanHierarchicalImplicitModel
    model = BunyaanHierarchicalImplicitModel(feature_dim=16, hidden_dim=64)
    volume = torch.rand(1, 1, 32, 32, 32)
    points = torch.rand(1, 21, 3) * 2 - 1
    output = model(volume, points)
    assert output["sdf"].shape == (1, 21, 1)
    assert output["base_color"].shape == (1, 21, 3)


def test_implicit_mesh_extraction_preserves_vertex_pbr_fields():
    import pytest
    pytest.importorskip("skimage")
    from torch import nn
    from bunyaan_model.architecture import TextConditioner
    from bunyaan_model.implicit_mesh import implicit_field_to_mesh

    class DummyDecoder(nn.Module):
        def forward(self, feature_grid, points, condition_tokens=None, condition_padding_mask=None):
            sdf = points.norm(dim=-1, keepdim=True) - .62
            normal = torch.nn.functional.normalize(points, dim=-1, eps=1e-6)
            n = points.shape[1]
            return {"sdf": sdf, "occupancy": torch.sigmoid(-sdf*32), "normal": normal,
                    "base_color": torch.full((1,n,3), .4, device=points.device),
                    "roughness": torch.full((1,n,1), .5, device=points.device),
                    "metallic": torch.full((1,n,1), .1, device=points.device),
                    "opacity": torch.ones((1,n,1), device=points.device)}

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__(); self.anchor = nn.Parameter(torch.zeros(())); self.decoder = DummyDecoder()
        def encoder(self, volume): return volume

    result = implicit_field_to_mesh(DummyModel(), TextConditioner(), torch.zeros(1,1,4,4,4),
                                    resolution=16, chunk_size=256)
    assert result["vertices"].ndim == 2 and result["vertices"].shape[1] == 3
    assert result["faces"].ndim == 2 and result["faces"].shape[1] == 3
    assert result["base_color"].shape == result["vertices"].shape
    assert result["roughness"].shape == (len(result["vertices"]), 1)


def test_implicit_checkpoint_loader_round_trip(tmp_path):
    from bunyaan_model.architecture import TextConditioner
    from bunyaan_model.implicit3d import BunyaanHierarchicalImplicitModel, load_implicit_checkpoint
    model = BunyaanHierarchicalImplicitModel(feature_dim=16, hidden_dim=64)
    text = TextConditioner(256)
    path = tmp_path / "implicit.pt"
    torch.save({"format":"bunyaan-implicit-field", "config":{"feature_dim":16,"hidden_dim":64,
                "condition_dim":256,"has_views":False}, "field_model":model.state_dict(),
                "text":text.state_dict(), "ema_field_model":model.state_dict(),
                "ema_text":text.state_dict()}, path)
    loaded, loaded_text, views, payload = load_implicit_checkpoint(path)
    assert views is None and payload["format"] == "bunyaan-implicit-field"
    assert loaded.encoder.down2[0].weight.shape[0] == 16
    assert len(loaded_text.encoder.layers) == 4
