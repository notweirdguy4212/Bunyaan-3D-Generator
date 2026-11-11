#!/usr/bin/env python3
"""Extract UV/PBR material samples from GLB files into Bunyaan five-channel tensors."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFile
import trimesh
ImageFile.LOAD_TRUNCATED_IMAGES = False


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def image_array(value, default, size):
    if value is None: return np.full((size,size,3), default, dtype=np.float32)
    if isinstance(value, Image.Image): im=value
    else:
        try: im=Image.fromarray(np.asarray(value))
        except Exception: return np.full((size,size,3), default, dtype=np.float32)
    im=im.convert('RGB').resize((size,size),Image.Resampling.LANCZOS)
    return np.asarray(im,dtype=np.float32)/255.

def scalar_factor(material, name, fallback):
    v=getattr(material,name,None) if material is not None else None
    try: return float(v) if v is not None else fallback
    except Exception: return fallback

def material_images(material, size):
    base=getattr(material,'baseColorTexture',None) if material is not None else None
    mr=getattr(material,'metallicRoughnessTexture',None) if material is not None else None
    base_arr=image_array(base,[1,1,1],size)
    mr_arr=image_array(mr,[255,255,255],size)
    # glTF metallic-roughness texture: roughness in G, metallic in B.
    rough=mr_arr[:,:,1:2] if mr is not None else np.full((size,size,1),scalar_factor(material,'roughnessFactor',1.0),np.float32)
    metal=mr_arr[:,:,2:3] if mr is not None else np.full((size,size,1),scalar_factor(material,'metallicFactor',0.0),np.float32)
    # Apply glTF scalar factors without altering source color texture.
    rough=np.clip(rough*scalar_factor(material,'roughnessFactor',1.0),0,1)
    metal=np.clip(metal*scalar_factor(material,'metallicFactor',0.0),0,1)
    factor=getattr(material,'baseColorFactor',None) if material is not None else None
    if factor is not None:
        f=np.asarray(factor,dtype=np.float32).reshape(-1)
        if len(f)>=3: base_arr=np.clip(base_arr*(f[:3]/255. if f[:3].max()>1 else f[:3]),0,1)
    return base_arr,rough,metal

def raster_uv(uv, faces, resolution=512):
    # Rasterize UV triangles; count>1 indicates texel overlap between distinct faces.
    count=np.zeros((resolution,resolution),np.uint16)
    for tri in faces:
        pts=[(float(uv[i,0]*(resolution-1)),float((1-uv[i,1])*(resolution-1))) for i in tri]
        mask=Image.new('1',(resolution,resolution),0); ImageDraw.Draw(mask).polygon(pts,fill=1)
        count += np.asarray(mask,dtype=np.uint16)
    return count

def main():
    p=argparse.ArgumentParser(description='Bunyaan TexVerse GLB ingestion and five-channel corpus builder')
    p.add_argument('--input',required=True,help='Directory containing .glb files')
    p.add_argument('--out',default='datasets/bunyaan-texverse')
    p.add_argument('--manifest',default='datasets/bunyaan-texverse/manifest.jsonl')
    p.add_argument('--resolution',type=int,default=512)
    p.add_argument('--model-id',default='',help='Optional TexVerse model ID; otherwise filename stem')
    p.add_argument('--license',required=True,help='Explicit license identifier copied to provenance; e.g. CC0 or CC-BY-4.0')
    p.add_argument('--license-url',default='')
    p.add_argument('--attribution',default='')
    p.add_argument('--allow-invalid-uv',action='store_true',help='Write samples with UV warnings; default skips invalid UV samples')
    p.add_argument('--min-texture-resolution',type=int,default=256)
    p.add_argument('--report',default='reports/bunyaan-texverse-extraction.json')
    a=p.parse_args()
    if a.resolution<16: p.error('--resolution must be >=16')
    root=Path(a.input); files=sorted(root.rglob('*.glb'))
    if not files: raise SystemExit(f'No GLB files found under {root}')
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); manifest=Path(a.manifest); manifest.parent.mkdir(parents=True,exist_ok=True)
    stats={'assets':0,'samples':0,'skipped':0,'warnings':0}
    asset_results=[]
    with manifest.open('w',encoding='utf8') as mf:
      for glb in files:
        model_id=a.model_id or glb.stem
        digest=sha256(glb)
        try: scene=trimesh.load(glb,force='scene',process=False)
        except Exception as e:
            stats['skipped']+=1; asset_results.append({'source_file':str(glb),'model_id':model_id,'valid':False,'issues':['glb_read_error:'+str(e)]}); print(f'[Bunyaan] ERROR {glb}: {e}'); continue
        geometries=list(scene.geometry.items()) if isinstance(scene,trimesh.Scene) else [(glb.stem,scene)]
        for gi,(gname,mesh) in enumerate(geometries):
            if not isinstance(mesh,trimesh.Trimesh) or len(mesh.faces)==0: continue
            uv=getattr(getattr(mesh,'visual',None),'uv',None)
            mat=getattr(getattr(mesh,'visual',None),'material',None)
            issues=[]
            uv_counts=np.zeros((a.resolution,a.resolution),np.uint16)
            if uv is None: issues.append('missing_uv')
            else:
                uv=np.asarray(uv,dtype=np.float32)
                if uv.ndim!=2 or uv.shape[1]!=2 or len(uv)!=len(mesh.vertices): issues.append('uv_shape_mismatch')
                elif not np.isfinite(uv).all(): issues.append('uv_non_finite')
                elif (uv<0).any() or (uv>1).any(): issues.append('uv_out_of_range')
                else:
                    uv_counts=raster_uv(uv,mesh.faces,a.resolution)
                    ov=int((uv_counts>1).sum())
                    if ov: issues.append(f'uv_overlap_pixels:{ov}')
            if issues and not a.allow_invalid_uv:
                stats['skipped']+=1; stats['warnings']+=len(issues); asset_results.append({'source_file':str(glb),'model_id':model_id,'mesh_name':str(gname),'valid':False,'issues':issues}); print(f'[Bunyaan] SKIP {glb.name}/{gname}: {", ".join(issues)}'); continue
            if uv is None or len(uv)!=len(mesh.vertices) or not np.isfinite(uv).all():
                stats['skipped']+=1; asset_results.append({'source_file':str(glb),'model_id':model_id,'mesh_name':str(gname),'valid':False,'issues':issues or ['invalid_uv_data']}); continue
            try:
                base_tex=getattr(mat,'baseColorTexture',None) if mat is not None else None
                if base_tex is None: issues.append('base_color_texture_missing_using_material_factor')
                mr_tex=getattr(mat,'metallicRoughnessTexture',None) if mat is not None else None
                for label,tex in (('base_color',base_tex),('metallic_roughness',mr_tex)):
                    if tex is not None:
                        try:
                            if min(tex.size)<a.min_texture_resolution: issues.append(f'{label}_resolution_below_{a.min_texture_resolution}')
                            # Force pixel decoding now so truncated/corrupt embedded images fail here.
                            tex.load()
                        except Exception as exc: raise ValueError(f'{label}_texture_corrupt: {exc}')
                base,rough,metal=material_images(mat,a.resolution)
                tensor=np.concatenate([base,rough,metal],axis=2).transpose(2,0,1).astype(np.float32)
                if tensor.shape!=(5,a.resolution,a.resolution) or not np.isfinite(tensor).all(): raise ValueError('invalid five-channel tensor')
                safe=f'{model_id}_{gi:04d}'.replace('/','_').replace('\\','_')
                sample=out/f'{safe}.pt'
                provenance={'brand':'Bunyaan','source':'TexVerse','source_file':str(glb.resolve()),'source_sha256':digest,
                    'model_id':model_id,'mesh_name':str(gname),'license':a.license,'license_url':a.license_url,
                    'attribution':a.attribution,'channels':['base_color_r','base_color_g','base_color_b','roughness','metallic'],
                    'color_space':'RGB sRGB; roughness/metallic linear','resolution':a.resolution,'validation_issues':issues}
                torch.save({'format':'bunyaan-pbr-tensor-v1','texture':torch.from_numpy(tensor),
                    'uv':torch.from_numpy(uv.copy()),'faces':torch.from_numpy(np.asarray(mesh.faces,dtype=np.int64)),
                    'uv_mask':torch.from_numpy((uv_counts>0).astype(np.float32)[None]),'caption':str(gname),'provenance':provenance},sample)
                mf.write(json.dumps({'sample':str(sample),'model_id':model_id,'source_sha256':digest,
                    'license':a.license,'license_url':a.license_url,'attribution':a.attribution,
                    'channels':provenance['channels'],'resolution':a.resolution,'validation_issues':issues},ensure_ascii=False)+'\n')
                stats['samples']+=1; stats['warnings']+=len(issues)
                asset_results.append({'source_file':str(glb),'model_id':model_id,'mesh_name':str(gname),'sample':str(sample),'valid':not bool(issues),'issues':issues})
            except Exception as e:
                stats['skipped']+=1; asset_results.append({'source_file':str(glb),'model_id':model_id,'mesh_name':str(gname),'valid':False,'issues':['extraction_error:'+str(e)]}); print(f'[Bunyaan] ERROR {glb.name}/{gname}: {e}')
        stats['assets']+=1
    summary={'brand':'Bunyaan','manifest':str(manifest),'output':str(out),**stats}
    report=Path(a.report); report.parent.mkdir(parents=True,exist_ok=True); report.write_text(json.dumps({**summary,'results':asset_results},indent=2),encoding='utf8')
    print(json.dumps({**summary,'report':str(report)},indent=2))
if __name__=='__main__': main()
