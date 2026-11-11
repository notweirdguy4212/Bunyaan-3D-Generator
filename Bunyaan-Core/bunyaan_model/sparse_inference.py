"""Sparse-coordinate inference adapter for Bunyaan's dense 3D denoiser.

Active voxels are grouped into local, padded tiles; only tiles containing active
coordinates are sent through the neural network. This is sparse tiled inference,
not a sparse-convolution kernel.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

@torch.inference_mode()
def infer_sparse_tiles(model, coordinates, features, tokens, tile_size=32, image=None):
    """Infer values at [batch,z,y,x] coordinates from local sparse tiles.

    coordinates: LongTensor [N,4], features: FloatTensor [N] or [N,1].
    Returns one predicted noise value per input coordinate in original order.
    """
    coords=torch.as_tensor(coordinates,dtype=torch.long)
    vals=torch.as_tensor(features,dtype=torch.float32,device=next(model.parameters()).device).reshape(-1)
    coords=coords.to(vals.device); tokens=tokens.to(vals.device)
    if coords.ndim!=2 or coords.shape[1]!=4 or len(coords)!=len(vals): raise ValueError('coordinates must be [N,4] and match features')
    if tile_size<8 or tile_size%8: raise ValueError('tile_size must be >=8 and divisible by 8')
    if not len(coords): return vals.new_empty((0,))
    if torch.any(coords<0): raise ValueError('Sparse coordinates must be non-negative')
    # Group by batch and tile; coordinate order is preserved in output.
    keys=torch.cat([coords[:,0:1],torch.div(coords[:,1:],tile_size,rounding_mode='floor')],1)
    unique,inverse=torch.unique(keys,dim=0,return_inverse=True)
    out=vals.new_zeros(len(vals))
    for group,key in enumerate(unique):
        ids=torch.where(inverse==group)[0]; batch=int(key[0]); origin=key[1:]*tile_size
        local=coords[ids,1:]-origin
        extent=int(local.max().item())+1
        side=max(tile_size,((extent+7)//8)*8)
        tile=vals.new_zeros((1,1,side,side,side))
        tile[0,0,local[:,0],local[:,1],local[:,2]]=vals[ids]
        t=torch.zeros(1,device=vals.device,dtype=torch.long)
        pred=model(tile,t,tokens[batch:batch+1],image[batch:batch+1] if image is not None else None)
        out[ids]=pred[0,0,local[:,0],local[:,1],local[:,2]]
    return out
