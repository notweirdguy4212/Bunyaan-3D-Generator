"""Optional FastAPI wrapper for a local Bunyaan checkpoint."""
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .service import BunyaanService
from .library import AssetLibrary
from .mesh_intelligence import analyze_mesh, process_mesh
from .materials import synthesize_pbr_maps
from .voxel import AdaptiveVoxelTree
from .multiview import reconstruct_from_silhouettes
from .sparse_inference import infer_sparse_tiles
from .surface import refine_surface, export_interchange
import tempfile, shutil

app=FastAPI(title="Bunyaan Native 3D API",version="1.2.0-dev",description="Independent Bunyaan 3D latent diffusion model")
service=None; library=AssetLibrary(os.getenv('BUNYAAN_DB','assets/bunyaan-library.sqlite3'))
class GenerateRequest(BaseModel):
    prompt:str=Field(min_length=1,max_length=2000)
    name:str=Field(default="Bunyaan generation",max_length=120)
    output_format:str=Field(default="glb",pattern="^(glb|gltf|obj|ply|stl)$")
    threshold:float=Field(default=0.5,ge=0.05,le=0.95)
    guidance:float=Field(default=1.0,ge=0.0,le=10.0)

@app.on_event('startup')
def startup():
    global service
    checkpoint=os.getenv('BUNYAAN_CHECKPOINT')
    if checkpoint and Path(checkpoint).exists(): service=BunyaanService(checkpoint,os.getenv('BUNYAAN_DEVICE'))

@app.get('/health')
def health():
    return {"status":"ok","model_loaded":service is not None,"model":"Bunyaan Native 3D",
            "capabilities":{"text_to_3d":service is not None,"image_to_3d":service is not None,
            "adaptive_voxel_tree":True,"mesh_analysis":True,"mesh_repair":True,
            "procedural_pbr_maps":True,"trained_texture_ai":Path(os.getenv("BUNYAAN_TEXTURE_CHECKPOINT","")).is_file(),
            "multi_view_reconstruction":True,"native_fbx_usd_export":True,"sparse_tiled_inference":True,
            "adaptive_surface_refinement":True}}

@app.post('/voxels/octree/stats')
async def voxel_octree_stats(file: UploadFile = File(...), tolerance: float = Form(0.0), min_size: int = Form(1)):
    # Accept NumPy .npy volumes only; pickle/object payloads are deliberately disabled.
    with tempfile.TemporaryDirectory() as td:
        src=Path(td)/'volume.npy'; src.write_bytes(await file.read())
        try:
            import numpy as np
            volume=np.load(src,allow_pickle=False)
            tree=AdaptiveVoxelTree(volume,tolerance=tolerance,min_size=min_size)
            return tree.stats()
        except Exception as e: raise HTTPException(400,str(e))

@app.post('/mesh/analyze')
async def mesh_analyze(file: UploadFile = File(...)):
    suffix=Path(file.filename or 'mesh.glb').suffix or '.glb'
    with tempfile.TemporaryDirectory() as td:
        src=Path(td)/('input'+suffix)
        src.write_bytes(await file.read())
        try: return analyze_mesh(src)
        except Exception as e: raise HTTPException(400,str(e))

@app.post('/mesh/process')
async def mesh_process(file: UploadFile = File(...), output_format: str = Form('glb'),
                       repair: bool = Form(True), target_faces: int | None = Form(None),
                       require_watertight: bool = Form(False)):
    allowed={'glb','gltf','obj','ply','stl'}
    if output_format not in allowed: raise HTTPException(400,'Supported formats: '+', '.join(sorted(allowed)))
    suffix=Path(file.filename or 'mesh.glb').suffix or '.glb'
    outdir=Path('assets/processed'); outdir.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        src=Path(td)/('input'+suffix); src.write_bytes(await file.read())
        dest=outdir/(str(__import__('uuid').uuid4())+'.'+output_format)
        try: result=process_mesh(src,dest,repair=repair,target_faces=target_faces,require_watertight=require_watertight)
        except Exception as e: raise HTTPException(400,str(e))
        return {**result,"download_url":"/processed/"+dest.name}

@app.post('/materials/synthesize')
def material_synthesize(prompt: str = Form('neutral material'), resolution: int = Form(1024),
                        base_color: str = Form('#9b7653'), metallic: float = Form(0.0),
                        roughness: float = Form(0.65)):
    outdir=Path('assets/materials')/str(__import__('uuid').uuid4())
    try:
        result=synthesize_pbr_maps(outdir,prompt,resolution,base_color,metallic,roughness)
        result['download_url']='/materials/'+outdir.name+'/download'
        return result
    except Exception as e: raise HTTPException(400,str(e))

@app.get('/processed/{filename}')
def download_processed(filename: str):
    # Restrict downloads to a single generated filename within the processed directory.
    if Path(filename).name != filename: raise HTTPException(400,'Invalid filename')
    path=(Path('assets/processed')/filename).resolve()
    if not path.is_file(): raise HTTPException(404,'File not found')
    return FileResponse(path,filename=path.name)

@app.get('/materials/{asset_id}/download')
def download_materials(asset_id: str):
    if Path(asset_id).name != asset_id: raise HTTPException(400,'Invalid asset id')
    path=(Path('assets/materials')/asset_id/'bunyaan-pbr-maps.zip').resolve()
    if not path.is_file(): raise HTTPException(404,'Material archive not found')
    return FileResponse(path,filename='bunyaan-pbr-maps.zip')


@app.post('/reconstruct/multiview')
async def multiview_reconstruct(file: UploadFile = File(...), resolution: int = Form(128)):
    # NPZ contains masks[V,H,W] and calibrated cameras[V,3,4].
    if resolution < 8 or resolution > 512: raise HTTPException(400,'resolution must be 8..512')
    with tempfile.TemporaryDirectory() as td:
        src=Path(td)/'views.npz'; src.write_bytes(await file.read())
        try:
            import numpy as np
            with np.load(src,allow_pickle=False) as data:
                volume=reconstruct_from_silhouettes(data['masks'],data['cameras'],resolution=resolution)
            dest=Path('assets/reconstructions'); dest.mkdir(parents=True,exist_ok=True)
            name=str(__import__('uuid').uuid4())+'.npy'; np.save(dest/name,volume)
            return {'download_url':'/reconstructions/'+name,'resolution':resolution,
                    'occupied_voxels':int(volume.sum()),'method':'calibrated multi-view visual hull'}
        except Exception as e: raise HTTPException(400,str(e))

@app.get('/reconstructions/{filename}')
def download_reconstruction(filename: str):
    if Path(filename).name!=filename: raise HTTPException(400,'Invalid filename')
    path=Path('assets/reconstructions')/filename
    if not path.is_file(): raise HTTPException(404,'Not found')
    return FileResponse(path,filename=path.name)

@app.post('/mesh/refine')
async def mesh_refine(file: UploadFile = File(...), target_edge_length: float = Form(0.03),
                      iterations: int = Form(3), smoothing: float = Form(0.25)):
    try:
        import trimesh
        suffix=Path(file.filename or 'mesh.glb').suffix or '.glb'
        with tempfile.TemporaryDirectory() as td:
            src=Path(td)/('input'+suffix); src.write_bytes(await file.read())
            loaded=trimesh.load(src,force='mesh',process=True)
            refined=refine_surface(loaded,target_edge_length,iterations,smoothing)
            destdir=Path('assets/refined'); destdir.mkdir(parents=True,exist_ok=True)
            dest=destdir/(str(__import__('uuid').uuid4())+'.glb'); refined.export(dest)
            return {'download_url':'/refined/'+dest.name,'vertices':len(refined.vertices),
                    'faces':len(refined.faces),'is_watertight':bool(refined.is_watertight)}
    except Exception as e: raise HTTPException(400,str(e))

@app.get('/refined/{filename}')
def download_refined(filename: str):
    if Path(filename).name!=filename: raise HTTPException(400,'Invalid filename')
    path=Path('assets/refined')/filename
    if not path.is_file(): raise HTTPException(404,'Not found')
    return FileResponse(path,filename=path.name)

@app.post('/mesh/export-bridge')
async def mesh_export_bridge(file: UploadFile = File(...), output_format: str = Form(...)):
    if output_format not in {'fbx','usd','usda','usdc'}: raise HTTPException(400,'Unsupported bridge format')
    suffix=Path(file.filename or 'mesh.glb').suffix or '.glb'
    try:
        with tempfile.TemporaryDirectory() as td:
            src=Path(td)/('input'+suffix); src.write_bytes(await file.read())
            destdir=Path('assets/exports'); destdir.mkdir(parents=True,exist_ok=True)
            dest=destdir/(str(__import__('uuid').uuid4())+'.'+output_format)
            result=export_interchange(src,dest,os.getenv('BLENDER_EXECUTABLE','blender'))
            return {**result,'download_url':'/exports/'+dest.name}
    except Exception as e: raise HTTPException(400,str(e))

@app.get('/exports/{filename}')
def download_bridge_export(filename: str):
    if Path(filename).name!=filename: raise HTTPException(400,'Invalid filename')
    path=Path('assets/exports')/filename
    if not path.is_file(): raise HTTPException(404,'Not found')
    return FileResponse(path,filename=path.name)

@app.post('/inference/sparse')
def sparse_inference(payload: dict):
    if service is None: raise HTTPException(503,'No Bunyaan checkpoint loaded')
    try:
        import torch
        coords=torch.tensor(payload['coordinates'],dtype=torch.long,device=service.device)
        values=torch.tensor(payload['values'],dtype=torch.float32,device=service.device)
        tokens=TextConditioner.tokenize([payload.get('prompt','')],device=service.device)
        result=infer_sparse_tiles(service.model,coords,values,tokens,tile_size=int(payload.get('tile_size',32)))
        return {'predictions':result.cpu().tolist(),'count':len(result),'mode':'sparse tiled neural inference'}
    except Exception as e: raise HTTPException(400,str(e))

@app.post('/materials/synthesize-learned')
def material_synthesize_learned(prompt: str = Form(...), resolution: int = Form(256), steps: int = Form(50)):
    checkpoint=os.getenv('BUNYAAN_TEXTURE_CHECKPOINT')
    if not checkpoint or not Path(checkpoint).is_file():
        raise HTTPException(503,'Set BUNYAAN_TEXTURE_CHECKPOINT to a trained bunyaan-texture-v1 checkpoint')
    try:
        from .texture_ai import load_texture_checkpoint, synthesize_learned_texture
        model,_=load_texture_checkpoint(checkpoint,os.getenv('BUNYAAN_DEVICE','cpu'))
        outdir=Path('assets/materials')/str(__import__('uuid').uuid4())
        result=synthesize_learned_texture(model,prompt,outdir,resolution,steps,os.getenv('BUNYAAN_DEVICE','cpu'))
        result['download_url']='/materials/'+outdir.name+'/download'
        return result
    except Exception as e: raise HTTPException(400,str(e))

@app.post('/generate')
def generate(req:GenerateRequest):
    if service is None: raise HTTPException(503,"No trained Bunyaan checkpoint loaded. Set BUNYAAN_CHECKPOINT to a native Bunyaan checkpoint.")
    outdir=Path('assets/generated'); outdir.mkdir(parents=True,exist_ok=True)
    path=outdir/(str(__import__('uuid').uuid4())+'.'+req.output_format)
    try:
        result=service.generate(req.prompt,path,threshold=req.threshold,guidance=req.guidance)
        meta={k:v for k,v in result.items() if k not in ('occupancy','path')}
        asset=library.add(req.name,req.prompt,path,meta)
        return {"asset":asset,"mesh":{k:v for k,v in result.items() if k!='occupancy'}}
    except Exception as e:
        if path.exists(): path.unlink()
        raise HTTPException(400,str(e))

@app.get('/assets')
def assets(q:str|None=None,limit:int=100): return library.list(min(max(limit,1),500),q)
@app.get('/assets/{asset_id}')
def asset(asset_id:str):
    item=library.get(asset_id)
    if not item: raise HTTPException(404,'Asset not found')
    return item
@app.delete('/assets/{asset_id}')
def delete_asset(asset_id:str):
    if not library.delete(asset_id): raise HTTPException(404,'Asset not found')
    return {"deleted":True,"id":asset_id}

@app.get('/',include_in_schema=False)
def home():
    return FileResponse(Path(__file__).resolve().parent.parent / 'web' / 'index.html')

@app.get('/assets/{asset_id}/download')
def download_asset(asset_id:str):
    item=library.get(asset_id)
    if not item: raise HTTPException(404,'Asset not found')
    path=Path(item['file_path']).resolve()
    if not path.is_file(): raise HTTPException(404,'Asset file not found')
    return FileResponse(path,filename=path.name,media_type='application/octet-stream')
