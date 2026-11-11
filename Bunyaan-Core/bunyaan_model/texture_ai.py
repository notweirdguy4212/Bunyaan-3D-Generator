"""Trainable neural PBR texture synthesis.

This module defines a conditional denoising network and inference/training-ready
interfaces. It does not ship pretrained weights: useful learned output requires
training on licensed, UV-aligned material datasets.
"""
from __future__ import annotations
from pathlib import Path
import json
import torch
from torch import nn
import torch.nn.functional as F
from .architecture import TextConditioner
from .attention import SpatialConditionAttention, LinearSpatialAttention

class TextureResidual(nn.Module):
    def __init__(self, channels, cond):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(8 if channels % 8 == 0 else 1, channels),
            nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8 if channels % 8 == 0 else 1, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1))
        self.film = nn.Linear(cond, channels * 2)
    def forward(self, x, c):
        scale, shift = self.film(c).chunk(2, 1)
        h = self.net[0](x)
        h = h * (1 + scale[:,:,None,None]) + shift[:,:,None,None]
        h = self.net[1:](h)
        return x + h * 0.2

class BunyaanTextureNet(nn.Module):
    """Predicts 5-channel UV PBR maps (RGB base color + roughness + metallic)."""
    def __init__(self, width=64, cond_dim=256, text_layers=2):
        super().__init__()
        self.text = TextConditioner(cond_dim, layers=text_layers)
        self.time = nn.Sequential(nn.Linear(cond_dim,cond_dim),nn.SiLU(),nn.Linear(cond_dim,cond_dim))
        self.inp = nn.Conv2d(5,width,3,padding=1)
        self.down1=nn.Conv2d(width,width*2,4,2,1); self.rb1=TextureResidual(width*2,cond_dim)
        self.down2=nn.Conv2d(width*2,width*4,4,2,1); self.rb2=TextureResidual(width*4,cond_dim)
        self.mid=TextureResidual(width*4,cond_dim); self.mid2=TextureResidual(width*4,cond_dim)
        self.text_attention=SpatialConditionAttention(width*4,cond_dim,heads=8)
        self.global_attention=LinearSpatialAttention(width*4,heads=8)
        self.up1=nn.ConvTranspose2d(width*4,width*2,4,2,1); self.mix1=nn.Conv2d(width*4,width*2,1); self.ur1=TextureResidual(width*2,cond_dim)
        self.up2=nn.ConvTranspose2d(width*2,width,4,2,1); self.mix2=nn.Conv2d(width*2,width,1); self.ur2=TextureResidual(width,cond_dim)
        self.out=nn.Sequential(nn.GroupNorm(8 if width%8==0 else 1,width),nn.SiLU(),nn.Conv2d(width,5,3,padding=1))
    def forward(self, noisy, timestep, tokens):
        context,padding=self.text.encode_tokens(tokens)
        c=self.text(tokens)+self.time(self._time(timestep,self.text.token.embedding_dim))
        x=self.inp(noisy); a=self.rb1(self.down1(x),c); b=self.rb2(self.down2(a),c)
        x=self.mid2(self.mid(b,c),c); x=self.text_attention(x,context,padding); x=self.global_attention(x); x=self.ur1(self.mix1(torch.cat([self.up1(x),a],1)),c)
        x=self.ur2(self.mix2(torch.cat([self.up2(x),self.inp(noisy)],1)),c)
        return self.out(x)
    @staticmethod
    def _time(t,dim):
        half=dim//2; freq=torch.exp(-torch.log(torch.tensor(10000.,device=t.device))*torch.arange(half,device=t.device).float()/max(half-1,1))
        phase=t.float()[:,None]*freq[None,:]
        return F.pad(torch.cat([phase.sin(),phase.cos()],1),(0,dim-2*half))

def synthesize_learned_texture(model, prompt, output_dir, resolution=1024, steps=50, device=None, guidance=3.0, seed=42):
    """Run a trained texture denoiser and save interoperable PBR maps."""
    if resolution < 64 or resolution % 4: raise ValueError("resolution must be >=64 and divisible by 4")
    if steps < 1: raise ValueError("steps must be positive")
    device=torch.device(device or next(model.parameters()).device); model=model.to(device).eval()
    tokens=TextConditioner.tokenize([prompt],device=device)
    generator=torch.Generator(device=device).manual_seed(int(seed))
    x=torch.randn(1,5,resolution,resolution,device=device,generator=generator)
    # Linear beta DDIM schedule, matching the noise-prediction training objective.
    betas=torch.linspace(1e-4,0.02,1000,device=device)
    alpha_bar=torch.cumprod(1-betas,0)
    indices=torch.linspace(999,0,steps,device=device).long()
    with torch.inference_mode():
        for i,timestep in enumerate(indices):
            ts=timestep.expand(1)
            eps=model(x,ts,tokens)
            if guidance != 1.0:
                null=torch.zeros_like(tokens); null[:,0]=1
                eps0=model(x,ts,null)
                eps=eps0+guidance*(eps-eps0)
            a=alpha_bar[timestep]
            x0=(x-(1-a).sqrt()*eps)/a.sqrt().clamp_min(1e-5)
            if i+1<len(indices):
                prev=alpha_bar[indices[i+1]]
                x=prev.sqrt()*x0+(1-prev).sqrt()*eps
            else:
                x=x0
        maps=x.sigmoid()[0].cpu()
    from PIL import Image
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    rgb=(maps[:3].permute(1,2,0).numpy()*255).round().astype('uint8')
    rough=(maps[3].numpy()*255).round().astype('uint8'); metal=(maps[4].numpy()*255).round().astype('uint8')
    import numpy as np
    gy,gx=np.gradient(rough.astype(np.float32)/255)
    normal=np.stack([128-gx*32,128-gy*32,np.full_like(gx,250)],-1).clip(0,255).astype('uint8')
    arrays={'base_color':rgb,'roughness':rough,'metallic':metal,'normal':normal,
            'ambient_occlusion':np.full((resolution,resolution),235,dtype=np.uint8)}
    paths={}
    for name,array in arrays.items():
        path=out/f'{name}.png'; Image.fromarray(array).save(path,optimize=True); paths[name]=str(path)
    manifest={'generator':'BunyaanTextureNet','trained_ai':bool(getattr(model,'_bunyaan_trained',False)),'checkpoint_required':True,
              'prompt':prompt,'resolution':resolution,'guidance':guidance,'seed':int(seed),'maps':paths,
              'channel_order':['base_color_r','base_color_g','base_color_b','roughness','metallic'],
              'normal_map_note':'Derived from roughness gradients; not a separately learned normal channel.',
              'ao_map_note':'Neutral placeholder; not a separately learned ambient-occlusion channel.'}
    (out/'material.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    import zipfile
    archive=out/'bunyaan-pbr-maps.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
        for item in list(out.glob('*.png'))+[out/'material.json']: bundle.write(item,item.name)
    manifest['archive']=str(archive)
    return manifest

def load_texture_checkpoint(path, device='cpu', use_ema=True):
    payload=torch.load(path,map_location=device,weights_only=True)
    if payload.get('format') not in ('bunyaan-texture-v1','bunyaan-texture-diffusion-v2'): raise ValueError('Not a Bunyaan texture checkpoint')
    config=payload.get('config',{})
    model_config={k:config[k] for k in ('width','cond_dim','text_layers') if k in config}
    model=BunyaanTextureNet(**model_config); state=payload.get('ema') if use_ema else None
    model.load_state_dict(state or payload['model'], strict=False)
    model._bunyaan_trained = True
    return model.to(device).eval(),payload
