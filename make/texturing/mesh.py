"""Triangle soup of the mesh being textured, as exported from Blender."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MeshData:
    """World-space triangles with their UVs.

    positions: (T, 3, 3) corners per triangle.
    uvs: (T, 3, 2) UV per corner, Blender convention (v up, 0..1).
    normals: (T, 3) unit face normals, pointing out of the surface.
    """

    positions: np.ndarray
    uvs: np.ndarray
    normals: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "MeshData":
        with np.load(path) as data:
            return cls(positions=data["positions"].astype(np.float64), uvs=data["uvs"].astype(np.float64),
                       normals=data["normals"].astype(np.float64))

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        points = self.positions.reshape(-1, 3)
        return points.min(axis=0), points.max(axis=0)
