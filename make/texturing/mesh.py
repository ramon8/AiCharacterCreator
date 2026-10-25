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

    def corner_normals(self, smooth_angle: float) -> np.ndarray:
        """(T, 3, 3) per-corner normals, smoothed across edges flatter than
        `smooth_angle` degrees: each corner averages the face normals of the
        triangles that share its position and are within that angle of its own face."""
        corners = self.positions.reshape(-1, 3)
        face_of = np.repeat(np.arange(len(self.positions)), 3)
        _, group = np.unique(np.round(corners, 6), axis=0, return_inverse=True)
        group = group.ravel()
        limit = np.cos(np.radians(smooth_angle))
        out = np.repeat(self.normals, 3, axis=0)
        order = np.argsort(group, kind="stable")
        bounds = np.flatnonzero(np.diff(group[order])) + 1
        for members in np.split(order, bounds):
            normals = self.normals[face_of[members]]
            close = normals @ normals.T >= limit - 1e-9
            summed = close.astype(float) @ normals
            out[members] = summed / np.linalg.norm(summed, axis=1, keepdims=True)
        return out.reshape(-1, 3, 3)
