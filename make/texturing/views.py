"""Projection cameras. One `View` is one orthographic camera around the mesh.

The camera math lives here only: Blender gets the resulting `matrix_world`
and `ortho_scale`, and the projection uses `project()`, so the renders that
condition the generation and the projection of its output always agree.

Convention: the character faces -Y (Blender's front view). Azimuth 0 puts
the camera in front (on -Y) looking at +Y; azimuth 90 puts it on +X. Positive
elevation raises the camera.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np

from make.texturing.mesh import MeshData

# What each named camera shows, for view-dependent prompting: telling the model
# which side it paints keeps the back from getting a second face or chest emblem.
SIDES = {"front": "front view", "right": "side view from the right", "left": "side view from the left",
         "back": "back view, seen from behind", "top": "top view, seen from above"}


@dataclass(frozen=True)
class View:
    name: str
    target: tuple[float, float, float]
    azimuth: float
    elevation: float
    ortho_scale: float
    resolution: int = 1024
    distance: float = 10.0

    @classmethod
    def framing(cls, mesh: MeshData, name: str = "front", azimuth: float = 0.0, elevation: float = 0.0,
                resolution: int = 1024, margin: float = 1.1) -> "View":
        """A view centered on the mesh bounding box that fits it with some margin."""
        lo, hi = mesh.bounds()
        center = (lo + hi) / 2
        probe = cls(name, tuple(center), azimuth, elevation, 1.0, resolution)
        right, up, _ = probe.basis()
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        offsets = corners - center
        extent = 2 * max(np.abs(offsets @ right).max(), np.abs(offsets @ up).max())
        radius = float(np.linalg.norm(hi - lo))
        return cls(name, tuple(float(c) for c in center), azimuth, elevation, float(extent * margin),
                   resolution, distance=max(10.0, 2 * radius))

    @classmethod
    def around(cls, mesh: MeshData, angles: Iterable[tuple[str, float, float]],
               resolution: int = 1024) -> list["View"]:
        """One framing view per (name, azimuth, elevation), in that order."""
        return [cls.framing(mesh, name=name, azimuth=az, elevation=el, resolution=resolution)
                for name, az, el in angles]

    @property
    def side(self) -> str | None:
        """Prompt phrase naming the side of the character this view shows (None for
        views without one, such as the in-between preview angles)."""
        return SIDES.get(self.name)

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(right, up, forward) unit vectors; forward points from the camera into the scene."""
        az, el = math.radians(self.azimuth), math.radians(self.elevation)
        to_camera = np.array([math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)])
        forward = -to_camera
        right = np.cross(forward, (0.0, 0.0, 1.0))
        if np.linalg.norm(right) < 1e-9:  # looking straight up or down
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        return right, up, forward

    def position(self) -> np.ndarray:
        return np.asarray(self.target) - self.basis()[2] * self.distance

    def matrix_world(self) -> list[list[float]]:
        """Blender camera matrix: local X = right, Y = up, -Z = forward."""
        right, up, forward = self.basis()
        m = np.eye(4)
        m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, up, -forward, self.position()
        return m.tolist()

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World points (..., 3) -> pixel x, pixel y (from the top), depth along the view."""
        right, up, forward = self.basis()
        rel = np.asarray(points) - np.asarray(self.target)
        x = (rel @ right / self.ortho_scale + 0.5) * self.resolution
        y = (0.5 - rel @ up / self.ortho_scale) * self.resolution
        depth = rel @ forward + self.distance
        return x, y, depth

    def depth_range(self, mesh: MeshData) -> tuple[float, float]:
        _, _, depth = self.project(mesh.positions.reshape(-1, 3))
        return float(depth.min()), float(depth.max())

    def to_dict(self) -> dict:
        return asdict(self)
