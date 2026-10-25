"""Image generation client: conditions -> image.

`GenerationClient` is the one interface the texturer depends on. Two
implementations: `ComfyUIClient` (real, ComfyUI HTTP API) and
`FakeGenerationClient` (fixed images, no GPU) for tests.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlencode

from PIL import Image

from make import settings
from make.comfyui import http
from make.generation.workflow import Workflow


@dataclass(frozen=True)
class Conditions:
    """Everything a generation is conditioned on.

    depth/normal: renders of our mesh from the projection camera; the output
        has their size (must be a multiple of 8, ~1024 px is SDXL's sweet spot).
    references: reference images for IP-Adapter, one per workflow slot.
    mask: optional; white = object. The output gets it as alpha channel.
    """

    depth: Image.Image
    normal: Image.Image
    prompt: str
    seed: int
    references: tuple[Image.Image, ...] = ()
    mask: Image.Image | None = None
    negative_prompt: str = "blurry, lowres, text, watermark, signature, jpeg artifacts"

    @classmethod
    def from_files(cls, depth: Path, normal: Path, prompt: str, seed: int,
                   references: tuple[Path, ...] = (), mask: Path | None = None, **kwargs) -> "Conditions":
        return cls(
            depth=Image.open(depth), normal=Image.open(normal), prompt=prompt, seed=seed,
            references=tuple(Image.open(r) for r in references),
            mask=Image.open(mask) if mask else None, **kwargs,
        )


class GenerationClient(Protocol):
    def generate(self, conditions: Conditions) -> Image.Image: ...


class GenerationError(Exception):
    pass


def _apply_mask(image: Image.Image, mask: Image.Image | None) -> Image.Image:
    if mask is None:
        return image
    image = image.convert("RGBA")
    image.putalpha(mask.convert("L").resize(image.size, Image.Resampling.NEAREST))
    return image


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


class FakeGenerationClient:
    """Returns a fixed image (solid color, or a given one) at the render size."""

    def __init__(self, image: Image.Image | None = None, color: tuple[int, int, int] = (128, 96, 64)):
        self.image = image
        self.color = color
        self.calls: list[Conditions] = []

    def generate(self, conditions: Conditions) -> Image.Image:
        self.calls.append(conditions)
        size = conditions.depth.size
        image = self.image.convert("RGB").resize(size) if self.image else Image.new("RGB", size, self.color)
        return _apply_mask(image, conditions.mask)


class ComfyUIClient:
    def __init__(self, url: str | None = None, workflow: Workflow | None = None,
                 work_dir: Path | None = None, timeout: float = 600, poll_interval: float = 0.25):
        self.url = (url or settings.comfyui_url()).rstrip("/")
        self.workflow = workflow or Workflow.load_default()
        self.work_dir = Path(work_dir or settings.work_dir()) / "generations"
        self.timeout = timeout
        self.poll_interval = poll_interval

    def generate(self, conditions: Conditions) -> Image.Image:
        width, height = conditions.depth.size
        if width % 8 or height % 8:
            raise ValueError(f"render size {width}x{height} must be a multiple of 8")
        if conditions.normal.size != conditions.depth.size:
            raise ValueError("depth and normal renders must have the same size")
        slots = self.workflow.reference_slots()
        if len(conditions.references) != slots:
            raise ValueError(
                f"workflow {self.workflow.name!r} takes {slots} reference image(s), got {len(conditions.references)}"
            )

        values: dict[str, dict] = {
            "input:prompt": {"text": conditions.prompt},
            "input:negative": {"text": conditions.negative_prompt},
            "input:depth": {"image": self._upload(conditions.depth.convert("RGB"))},
            "input:normal": {"image": self._upload(conditions.normal.convert("RGB"))},
            "input:size": {"width": width, "height": height},
            "input:seed": {"seed": conditions.seed},
        }
        for index, reference in enumerate(conditions.references):
            values[f"input:reference:{index}"] = {"image": self._upload(reference.convert("RGB"))}
        graph = self.workflow.bind(values)

        image = self._run(graph)
        image = _apply_mask(image, conditions.mask)
        self._save(graph, image)
        return image

    def free(self, unload_models: bool = False) -> None:
        """Drop ComfyUI's cached node outputs (and optionally unload models from VRAM).

        ComfyUI reuses cached outputs for an identical graph, so without this a
        repeated generation returns instantly without recomputing.
        """
        http.post_json(f"{self.url}/free", {"free_memory": True, "unload_models": unload_models})

    def _upload(self, image: Image.Image) -> str:
        data = _png(image)
        name = f"acc_{hashlib.sha1(data).hexdigest()[:16]}.png"
        response = http.post_file(f"{self.url}/upload/image", "image", name, data,
                                  {"overwrite": "true", "type": "input"})
        return response["name"]

    def _run(self, graph: dict) -> Image.Image:
        try:
            queued = http.post_json(f"{self.url}/prompt", {"prompt": graph, "client_id": uuid.uuid4().hex})
        except http.HttpError as error:
            raise GenerationError(f"ComfyUI rejected workflow {self.workflow.name!r}: {error.body}") from error
        prompt_id = queued["prompt_id"]

        deadline = time.monotonic() + self.timeout
        while True:
            history = http.get_json(f"{self.url}/history/{prompt_id}")
            if prompt_id in history:
                entry = history[prompt_id]
                break
            if time.monotonic() > deadline:
                raise GenerationError(f"ComfyUI prompt {prompt_id} did not finish in {self.timeout}s")
            time.sleep(self.poll_interval)

        status = entry.get("status", {})
        if status.get("status_str") == "error":
            raise GenerationError(f"ComfyUI failed prompt {prompt_id}: {json.dumps(status.get('messages'))}")
        images = entry["outputs"][self.workflow.node_id("output:image")]["images"]
        ref = images[0]
        query = urlencode({"filename": ref["filename"], "subfolder": ref["subfolder"], "type": ref["type"]})
        data = http.get_bytes(f"{self.url}/view?{query}")
        return Image.open(io.BytesIO(data)).convert("RGB")

    def _save(self, graph: dict, image: Image.Image) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{time.strftime('%Y%m%d-%H%M%S')}_{self.workflow.name}_{uuid.uuid4().hex[:6]}"
        image.save(self.work_dir / f"{stem}.png")
        (self.work_dir / f"{stem}.json").write_text(json.dumps(graph, indent=1), encoding="utf-8")
