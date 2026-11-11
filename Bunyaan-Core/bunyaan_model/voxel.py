"""Adaptive hierarchical voxel representation for Bunyaan volumes.

The tree stores homogeneous regions as single nodes and subdivides mixed regions.
It is a compact CPU-side representation; it does not imply sparse neural inference.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class OctreeNode:
    origin: tuple[int, int, int]
    size: int
    value: float | None = None
    children: list["OctreeNode"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children

class AdaptiveVoxelTree:
    def __init__(self, volume, tolerance: float = 0.0, min_size: int = 1):
        a = np.asarray(volume, dtype=np.float32).squeeze()
        if a.ndim != 3 or len(set(a.shape)) != 1:
            raise ValueError("AdaptiveVoxelTree expects a cubic [R,R,R] volume")
        if not np.isfinite(a).all(): raise ValueError("Volume contains NaN/Inf")
        if tolerance < 0 or min_size < 1: raise ValueError("Invalid tolerance/min_size")
        self.volume = a
        self.tolerance = float(tolerance)
        self.min_size = int(min_size)
        self.root = self._build((0, 0, 0), a.shape[0])

    def _build(self, origin, size):
        x,y,z = origin; block = self.volume[x:x+size,y:y+size,z:z+size]
        lo, hi = float(block.min()), float(block.max())
        if hi-lo <= self.tolerance or size <= self.min_size:
            return OctreeNode(origin, size, float(block.mean()))
        half = size // 2
        if half == 0: return OctreeNode(origin, size, float(block.mean()))
        node = OctreeNode(origin, size)
        for dx in (0, half):
            for dy in (0, half):
                for dz in (0, half):
                    node.children.append(self._build((x+dx,y+dy,z+dz), half))
        return node

    def stats(self):
        leaves = 0; nodes = 0; max_depth = 0
        stack=[(self.root,0)]
        while stack:
            n,d=stack.pop(); nodes+=1; max_depth=max(max_depth,d)
            if n.is_leaf: leaves+=1
            else: stack.extend((c,d+1) for c in n.children)
        dense_bytes=self.volume.nbytes
        # Approximate compact payload: origin/size/value per leaf; Python object overhead excluded.
        return {"resolution": int(self.volume.shape[0]), "nodes": nodes, "leaves": leaves,
                "max_depth": max_depth, "dense_bytes": dense_bytes,
                "leaf_payload_bytes_estimate": leaves * 24,
                "compression_ratio_estimate": round(dense_bytes/max(1,leaves*24), 3)}

    def to_dense(self):
        out=np.zeros_like(self.volume)
        stack=[self.root]
        while stack:
            n=stack.pop(); x,y,z=n.origin; s=n.size
            if n.is_leaf: out[x:x+s,y:y+s,z:z+s]=n.value
            else: stack.extend(n.children)
        return out
