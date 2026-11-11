"""UV-space procedural PBR map authoring starter.

This is deterministic procedural synthesis, not a trained texture-generation model.
It emits interoperable base-color, roughness, metallic, normal and AO image maps.
"""
from pathlib import Path
import hashlib, zipfile
import numpy as np
from PIL import Image, ImageFilter

def _seed(text): return int.from_bytes(hashlib.sha256(text.encode('utf8')).digest()[:8], 'little')

def synthesize_pbr_maps(output_dir, prompt, resolution=1024, base_color="#9b7653", metallic=0.0, roughness=0.65):
    if resolution not in (1024,2048,4096): raise ValueError("resolution must be 1024, 2048 or 4096")
    if not 0<=metallic<=1 or not 0<=roughness<=1: raise ValueError("metallic and roughness must be in [0,1]")
    try: rgb=tuple(int(base_color.lstrip('#')[i:i+2],16) for i in (0,2,4))
    except Exception as e: raise ValueError("base_color must be a hex color such as #9b7653") from e
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(_seed(prompt+base_color+str(resolution)))
    # Generate at a reduced working resolution then upscale with smooth filtering to limit peak RAM.
    side=min(resolution,1024); noise=rng.normal(0,1,(side,side)).astype(np.float32)
    noise_img=Image.fromarray(np.uint8(np.clip(noise*24+128,0,255))).filter(ImageFilter.GaussianBlur(max(0.5,side/512)))
    n=(np.asarray(noise_img,dtype=np.float32)-128)/128
    base=np.zeros((side,side,3),dtype=np.uint8)
    for c in range(3): base[:,:,c]=np.clip(rgb[c]*(1+n*0.12),0,255)
    gray=np.uint8(np.clip(roughness*255+n*3,0,255))
    metal=np.full((side,side),round(metallic*255),dtype=np.uint8)
    # Neutral tangent-space normal with subtle height-derived perturbation.
    gy,gx=np.gradient(n); normal=np.stack([128-gx*18,128-gy*18,np.full_like(n,250)],axis=-1)
    normal=np.uint8(np.clip(normal,0,255)); ao=np.uint8(np.clip(235+n*12,0,255))
    maps={"base_color":Image.fromarray(base),"roughness":Image.fromarray(gray),
          "metallic":Image.fromarray(metal),"normal":Image.fromarray(normal),"ambient_occlusion":Image.fromarray(ao)}
    paths={}
    for name,img in maps.items():
        if side!=resolution: img=img.resize((resolution,resolution),Image.Resampling.LANCZOS)
        p=out/f"{name}.png"; img.save(p,optimize=True); paths[name]=str(p)
    manifest={"generator":"Bunyaan procedural UV-map starter","trained_ai":False,"resolution":resolution,
              "prompt":prompt,"base_color":base_color,"metallic":metallic,"roughness":roughness,"maps":paths}
    import json
    (out/'material.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    archive=out/'bunyaan-pbr-maps.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in list(out.glob('*.png'))+[out/'material.json']: z.write(p,p.name)
    manifest['archive']=str(archive)
    return manifest
