"""Voxelize a directory of licensed OBJ/PLY/STL/GLB assets into Bunyaan samples.

Provide captions.csv with columns: filename,caption. All source assets must be
owned, public-domain, or explicitly licensed for model training.
"""
import argparse,csv
from pathlib import Path
import numpy as np
import torch

def main():
 p=argparse.ArgumentParser();p.add_argument('--meshes',required=True);p.add_argument('--captions',required=True);p.add_argument('--out',required=True);p.add_argument('--resolution',type=int,default=32);a=p.parse_args()
 try: import trimesh
 except ImportError as e: raise SystemExit('Install trimesh: pip install trimesh') from e
 captions={}
 with open(a.captions,newline='',encoding='utf-8') as f:
  for row in csv.DictReader(f): captions[row['filename']]=row['caption']
 src=Path(a.meshes); out=Path(a.out);out.mkdir(parents=True,exist_ok=True); count=0
 for path in sorted(src.rglob('*')):
  if path.suffix.lower() not in {'.obj','.ply','.stl','.glb','.gltf'}:continue
  if path.name not in captions: print('skip (no caption):',path.name);continue
  mesh=trimesh.load(path,force='mesh')
  if not isinstance(mesh,trimesh.Trimesh) or len(mesh.vertices)==0: print('skip (not a mesh):',path);continue
  mesh=mesh.copy(); mesh.remove_unreferenced_vertices()
  ext=mesh.bounding_box.extents.max()
  if ext<=0:continue
  mesh.apply_translation(-mesh.bounding_box.centroid);mesh.apply_scale(0.9/ext)
  try:
   vox=mesh.voxelized(pitch=1.8/a.resolution).fill()
   matrix=vox.matrix.astype(np.float32)
   # Center/crop/pad into fixed cubic grid.
   target=np.zeros((a.resolution,)*3,np.float32)
   starts=[max(0,(a.resolution-s)//2) for s in matrix.shape]
   src_starts=[max(0,(s-a.resolution)//2) for s in matrix.shape]
   lengths=[min(a.resolution,s) for s in matrix.shape]
   target[starts[0]:starts[0]+lengths[0],starts[1]:starts[1]+lengths[1],starts[2]:starts[2]+lengths[2]]=matrix[src_starts[0]:src_starts[0]+lengths[0],src_starts[1]:src_starts[1]+lengths[1],src_starts[2]:src_starts[2]+lengths[2]]
   torch.save({'occupancy':torch.from_numpy(target).unsqueeze(0),'caption':captions[path.name]},out/(path.stem+'.pt'));count+=1
  except Exception as e: print('voxelization failed:',path,e)
 print(f'Prepared {count} samples in {out}. Verify data rights and inspect voxel quality before training.')
if __name__=='__main__':main()
