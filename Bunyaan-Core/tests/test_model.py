import torch
from bunyaan_model.architecture import Bunyaan3D, BunyaanDiffusion, TextConditioner

def test_text_tokenizer():
 t=TextConditioner.tokenize(['a red chair',''],max_length=16)
 assert t.shape==(2,16) and t[0,0].item()==1

def test_model_forward():
 model=Bunyaan3D(resolution=16,width=16,cond_dim=64,text_layers=1)
 tokens=TextConditioner.tokenize(['a small chair'],max_length=32)
 x=torch.randn(1,1,16,16,16); step=torch.tensor([4])
 y=model(x,step,tokens)
 assert y.shape==x.shape and torch.isfinite(y).all()

def test_diffusion():
 d=BunyaanDiffusion(steps=20); x=torch.zeros(2,1,8,8,8); n=torch.randn_like(x); t=torch.tensor([0,19])
 assert d.add_noise(x,n,t).shape==x.shape

def test_adaptive_voxel_tree_roundtrip():
 import numpy as np
 from bunyaan_model.voxel import AdaptiveVoxelTree
 volume=np.zeros((8,8,8),dtype=np.float32); volume[2:6,2:6,2:6]=1
 tree=AdaptiveVoxelTree(volume,tolerance=0,min_size=1)
 assert np.allclose(tree.to_dense(),volume)
 assert tree.stats()['nodes'] > 1

def test_procedural_pbr_maps(tmp_path):
 from bunyaan_model.materials import synthesize_pbr_maps
 result=synthesize_pbr_maps(tmp_path/'pbr','aged copper',1024,'#996644',0.8,0.35)
 assert set(result['maps']) == {'base_color','roughness','metallic','normal','ambient_occlusion'}
 assert (tmp_path/'pbr'/'bunyaan-pbr-maps.zip').is_file()

def test_mesh_inspection_and_repair(tmp_path):
 import trimesh
 from bunyaan_model.mesh_intelligence import analyze_mesh, process_mesh
 source=tmp_path/'box.glb'; target=tmp_path/'fixed.glb'
 trimesh.creation.box().export(source)
 assert analyze_mesh(source)['is_watertight'] is True
 result=process_mesh(source,target)
 assert result['is_watertight'] is True and target.is_file()
