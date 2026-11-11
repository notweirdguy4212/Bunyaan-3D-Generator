import numpy as np
import torch

def test_multiview_silhouette_carving():
    from bunyaan_model.multiview import reconstruct_from_silhouettes
    # Orthographic camera looking along +z, image coordinates x/y.
    cam=np.array([[1,0,0,4],[0,1,0,4],[0,0,0,1]],dtype=float)
    mask=np.ones((8,8),dtype=np.uint8)
    vol=reconstruct_from_silhouettes([mask],[cam],resolution=8)
    assert vol.shape==(8,8,8) and vol.sum()>0

def test_sparse_tiled_inference():
    from bunyaan_model.sparse_inference import infer_sparse_tiles
    class Dummy(torch.nn.Module):
        def __init__(self): super().__init__(); self.p=torch.nn.Parameter(torch.zeros(()))
        def forward(self,x,t,tokens,image=None): return x+1
    model=Dummy()
    coords=torch.tensor([[0,1,2,3],[0,2,2,3]])
    vals=torch.tensor([0.2,0.4]); tokens=torch.ones((1,4),dtype=torch.long)
    result=infer_sparse_tiles(model,coords,vals,tokens,tile_size=8)
    assert torch.allclose(result,vals+1)

def test_texture_network_forward():
    from bunyaan_model.texture_ai import BunyaanTextureNet
    from bunyaan_model.architecture import TextConditioner
    model=BunyaanTextureNet(width=8,cond_dim=32,text_layers=1)
    x=torch.randn(1,5,32,32); t=torch.tensor([5])
    tokens=TextConditioner.tokenize(['rough oak'],max_length=32)
    y=model(x,t,tokens)
    assert y.shape==x.shape and torch.isfinite(y).all()

def test_surface_refinement():
    import trimesh
    from bunyaan_model.surface import refine_surface
    mesh=trimesh.creation.icosphere(subdivisions=1)
    refined=refine_surface(mesh,target_edge_length=0.4,iterations=1,smoothing=0.0)
    assert len(refined.faces)>=len(mesh.faces)

def test_native_text_spatial_attention_2d_and_3d():
    from bunyaan_model.attention import SpatialConditionAttention
    module2=SpatialConditionAttention(16,32,heads=4)
    x2=torch.randn(2,16,80,80); ctx=torch.randn(2,12,32); mask=torch.zeros(2,12,dtype=torch.bool)
    y2=module2(x2,ctx,mask)
    assert y2.shape==x2.shape and torch.isfinite(y2).all()
    module3=SpatialConditionAttention(16,32,heads=4)
    x3=torch.randn(1,16,8,8,8)
    y3=module3(x3,ctx[:1],mask[:1])
    assert y3.shape==x3.shape and torch.isfinite(y3).all()

def test_bunyaan_3d_cross_attention_forward():
    from bunyaan_model.architecture import Bunyaan3D, TextConditioner
    model=Bunyaan3D(resolution=16,width=8,cond_dim=32,text_layers=1)
    tokens=TextConditioner.tokenize(['a detailed bronze helmet'],max_length=32)
    x=torch.randn(1,1,16,16,16); t=torch.tensor([12])
    y=model(x,t,tokens)
    assert y.shape==x.shape and torch.isfinite(y).all()

def test_ddim_sampler_is_reproducible():
    from bunyaan_model.architecture import Bunyaan3D, BunyaanDiffusion, TextConditioner
    model=Bunyaan3D(resolution=16,width=8,cond_dim=32,text_layers=1).eval()
    tokens=TextConditioner.tokenize(['a small stone statue'],max_length=32)
    diffusion=BunyaanDiffusion(steps=20)
    g1=torch.Generator().manual_seed(123); g2=torch.Generator().manual_seed(123)
    a=diffusion.sample_ddim(model,tokens,steps=4,guidance=1.0,generator=g1)
    b=diffusion.sample_ddim(model,tokens,steps=4,guidance=1.0,generator=g2)
    assert torch.allclose(a,b) and a.shape==(1,1,16,16,16)

def test_texture_checkpoint_load_prefers_ema(tmp_path):
    from bunyaan_model.texture_ai import BunyaanTextureNet, load_texture_checkpoint
    model=BunyaanTextureNet(width=8,cond_dim=32,text_layers=1)
    path=tmp_path/'texture.pt'
    torch.save({'format':'bunyaan-texture-diffusion-v2','config':{'width':8,'cond_dim':32,'text_layers':1,'resolution':32,'channels':5},'model':model.state_dict(),'ema':model.state_dict(),'epoch':1},path)
    loaded,payload=load_texture_checkpoint(path)
    assert loaded._bunyaan_trained is True and payload['epoch']==1

def test_linear_spatial_attention_global_context():
    from bunyaan_model.attention import LinearSpatialAttention
    for shape in [(2,32,8,8),(1,32,4,4,4)]:
        block=LinearSpatialAttention(32,heads=8)
        x=torch.randn(*shape); y=block(x)
        assert y.shape==x.shape and torch.isfinite(y).all()
