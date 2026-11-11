"""Mesh inspection and conservative repair utilities built on trimesh."""
from pathlib import Path
import numpy as np

def _trimesh():
    try: import trimesh
    except ImportError as e: raise RuntimeError("Mesh tools require trimesh: pip install trimesh") from e
    return trimesh

def load_mesh(path):
    tm=_trimesh(); obj=tm.load(str(path), force='mesh', process=False)
    if not isinstance(obj, tm.Trimesh): raise ValueError("Input did not contain a triangle mesh")
    return obj

def analyze_mesh(path):
    m=load_mesh(path)
    components=m.split(only_watertight=False)
    return {"vertices":int(len(m.vertices)),"faces":int(len(m.faces)),
        "is_watertight":bool(m.is_watertight),"is_winding_consistent":bool(m.is_winding_consistent),
        "is_volume":bool(m.is_volume),"euler_number":int(m.euler_number),
        "component_count":len(components),"bounds":m.bounds.tolist(),
        "extents":m.extents.tolist(),"surface_area":float(m.area),
        "volume":float(m.volume) if m.is_volume else None,
        "boundary_edge_count":int(len(m.edges_boundary)) if hasattr(m,'edges_boundary') else None}

def process_mesh(source, output, *, repair=True, target_faces=None, require_watertight=False):
    """Repair common defects, optionally simplify, then export to target extension.

    Hole filling is heuristic and can alter intended openings; inspect the result.
    Decimation requires the optional fast-simplification package.
    """
    m=load_mesh(source); before={"vertices":len(m.vertices),"faces":len(m.faces)}
    if repair:
        m.remove_infinite_values()
        # trimesh 4.x exposes face masks rather than the legacy removal methods.
        if hasattr(m, "unique_faces"): m.update_faces(m.unique_faces())
        elif hasattr(m, "remove_duplicate_faces"): m.remove_duplicate_faces()
        if hasattr(m, "nondegenerate_faces"): m.update_faces(m.nondegenerate_faces())
        elif hasattr(m, "remove_degenerate_faces"): m.remove_degenerate_faces()
        m.remove_unreferenced_vertices(); m.merge_vertices()
        trimesh=_trimesh()
        trimesh.repair.fix_winding(m); trimesh.repair.fix_normals(m)
        try: trimesh.repair.fill_holes(m)
        except Exception: pass
    if target_faces is not None:
        target_faces=int(target_faces)
        if target_faces < 4: raise ValueError("target_faces must be at least 4")
        if len(m.faces)>target_faces:
            try: m=m.simplify_quadric_decimation(face_count=target_faces)
            except Exception as e: raise RuntimeError("Decimation unavailable; install fast-simplification") from e
    if require_watertight and not m.is_watertight:
        raise ValueError("Mesh remains non-watertight after conservative repair")
    out=Path(output); out.parent.mkdir(parents=True,exist_ok=True); m.export(out)
    return {"path":str(out),"before":before,"after":{"vertices":len(m.vertices),"faces":len(m.faces)},
            "is_watertight":bool(m.is_watertight),"format":out.suffix.lower().lstrip('.')}
