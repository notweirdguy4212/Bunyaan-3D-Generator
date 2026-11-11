"""Surface quality refinement and Blender-backed interchange export bridges."""
from __future__ import annotations
from pathlib import Path
import os, shutil, subprocess, tempfile, json

def refine_surface(mesh, target_edge_length=None, iterations=5, smoothing=0.35, max_faces=2_000_000):
    """Adaptive edge-length subdivision followed by volume-preserving Taubin smoothing."""
    import trimesh
    if not isinstance(mesh,trimesh.Trimesh): raise TypeError('mesh must be a trimesh.Trimesh')
    if iterations<0 or not 0<=smoothing<=1: raise ValueError('Invalid refinement settings')
    result=mesh.copy()
    if len(result.faces)==0: raise ValueError('Mesh has no faces')
    if target_edge_length is not None:
        if target_edge_length<=0: raise ValueError('target_edge_length must be positive')
        # Split only edges longer than the target, bounded to avoid runaway memory.
        for _ in range(iterations):
            if len(result.faces)>=max_faces: break
            result=trimesh.remesh.subdivide_to_size(result.vertices,result.faces,
                max_edge=target_edge_length, max_iter=1)
            result=trimesh.Trimesh(vertices=result[0],faces=result[1],process=False)
    if smoothing and iterations:
        trimesh.smoothing.filter_taubin(result,lamb=smoothing,nu=-smoothing*1.05,iterations=iterations)
    result.remove_unreferenced_vertices()
    result.fix_normals()
    return result

def _blender_script(src,dst,fmt):
    return r"""import bpy, sys
from pathlib import Path
argv=sys.argv[sys.argv.index('--')+1:]
src,dst,fmt=argv
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.import_scene.gltf(filepath=src) if src.lower().endswith(('.glb','.gltf')) else bpy.ops.wm.import_mesh.stl(filepath=src) if src.lower().endswith('.stl') else bpy.ops.wm.import_scene.obj(filepath=src)
if fmt=='fbx': bpy.ops.export_scene.fbx(filepath=dst, use_selection=False, apply_unit_scale=True, bake_space_transform=False)
elif fmt=='usd': bpy.ops.wm.usd_export(filepath=dst, selected_objects_only=False, export_materials=True)
else: raise RuntimeError('Unsupported bridge format')
"""

def export_interchange(source, destination, blender='blender'):
    """Export an existing mesh file to FBX/USD using a real Blender installation."""
    src=Path(source).resolve(); dst=Path(destination).resolve()
    fmt=dst.suffix.lower()
    if fmt not in ('.fbx','.usd','.usda','.usdc'): raise ValueError('Destination must be .fbx, .usd, .usda or .usdc')
    exe=shutil.which(blender) if os.path.sep not in blender else blender
    if not exe or not Path(exe).is_file(): raise RuntimeError('Blender executable not found; install Blender and pass its executable path')
    if not src.is_file(): raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True,exist_ok=True)
    mode='fbx' if fmt=='.fbx' else 'usd'
    with tempfile.TemporaryDirectory() as td:
        script=Path(td)/'bridge.py'; script.write_text(_blender_script(str(src),str(dst),mode))
        run=subprocess.run([exe,'--background','--python',str(script),'--',str(src),str(dst),mode],
                           capture_output=True,text=True,timeout=600)
        if run.returncode or not dst.is_file():
            raise RuntimeError('Blender export failed: '+(run.stderr[-3000:] or run.stdout[-3000:]))
    return {'path':str(dst),'format':fmt[1:],'bridge':'Blender','bytes':dst.stat().st_size}
