"""Calibrated multi-view silhouette reconstruction by differentiable voxel carving."""
from __future__ import annotations
import numpy as np

def reconstruct_from_silhouettes(masks, cameras, resolution=128, bounds=((-0.5,-0.5,-0.5),(0.5,0.5,0.5)), chunk=250000):
    """Intersect binary silhouettes with calibrated 3x4 world-to-image projection matrices.

    masks: iterable HxW arrays; cameras: matching iterable 3x4 projection matrices.
    Returns float32 occupancy in [D,H,W]. This is geometric visual-hull reconstruction,
    not learned photometric stereo; camera calibration is the caller's responsibility.
    """
    masks=[np.asarray(m)>0 for m in masks]; cameras=[np.asarray(c,dtype=np.float64) for c in cameras]
    if not masks or len(masks)!=len(cameras): raise ValueError('Provide matching non-empty masks and cameras')
    if resolution<8 or any(c.shape!=(3,4) for c in cameras): raise ValueError('resolution >=8 and cameras must be 3x4')
    if any(m.ndim!=2 for m in masks): raise ValueError('Each silhouette must be a 2D mask')
    lo,hi=np.asarray(bounds[0]),np.asarray(bounds[1])
    if lo.shape!=(3,) or hi.shape!=(3,) or np.any(hi<=lo): raise ValueError('Invalid world bounds')
    axis=np.linspace(lo[0],hi[0],resolution,dtype=np.float32)
    yy=np.linspace(lo[1],hi[1],resolution,dtype=np.float32)
    zz=np.linspace(lo[2],hi[2],resolution,dtype=np.float32)
    grid=np.stack(np.meshgrid(axis,yy,zz,indexing='ij'),-1).reshape(-1,3)
    homogeneous=np.concatenate([grid,np.ones((len(grid),1),np.float32)],1)
    keep=np.ones(len(grid),dtype=bool)
    for mask,cam in zip(masks,cameras):
        h,w=mask.shape
        for start in range(0,len(grid),chunk):
            ids=np.flatnonzero(keep[start:start+chunk])+start
            if not len(ids): continue
            q=homogeneous[ids]@cam.T
            valid=q[:,2]>1e-8
            u=np.zeros(len(ids)); v=np.zeros(len(ids))
            u[valid]=q[valid,0]/q[valid,2]; v[valid]=q[valid,1]/q[valid,2]
            ui=np.rint(u).astype(int); vi=np.rint(v).astype(int)
            inside=valid&(ui>=0)&(ui<w)&(vi>=0)&(vi<h)
            hit=np.zeros(len(ids),dtype=bool); hit[inside]=mask[vi[inside],ui[inside]]
            keep[ids[~hit]]=False
    return keep.reshape(resolution,resolution,resolution).astype(np.float32)
