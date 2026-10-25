"""What the texturing pipeline needs from a ComfyUI install.

Single source of truth shared by the installer (what to download) and the
availability check (what to verify on a running server).
"""

from __future__ import annotations

from dataclasses import dataclass

# ComfyUI core, pinned so installs are reproducible. Override with
# COMFYUI_REF when deliberately upgrading.
COMFYUI_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
COMFYUI_REF = "19ae172bff6c9e4c738e7f2e00fe087f5b5b46ab"

TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"


@dataclass(frozen=True)
class Model:
    """A model file ComfyUI must list in `folder` under `filename`."""

    folder: str  # ComfyUI model folder name, e.g. "checkpoints"
    filename: str
    url: str


@dataclass(frozen=True)
class CustomNode:
    name: str
    repo: str
    ref: str
    node_types: tuple[str, ...]  # node classes it provides that we use


MODELS: tuple[Model, ...] = (
    Model(
        "checkpoints",
        "sd_xl_base_1.0.safetensors",
        "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
    ),
    Model(
        "controlnet",
        "controlnet-union-sdxl-promax.safetensors",
        "https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/main/diffusion_pytorch_model_promax.safetensors",
    ),
    Model(
        "ipadapter",
        "ip-adapter-plus_sdxl_vit-h.safetensors",
        "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors",
    ),
    Model(
        "clip_vision",
        "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
        "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors",
    ),
)

CUSTOM_NODES: tuple[CustomNode, ...] = (
    CustomNode(
        "ComfyUI_IPAdapter_plus",
        "https://github.com/cubiq/ComfyUI_IPAdapter_plus.git",
        "a0f451a5113cf9becb0847b92884cb10cbdec0ef",
        ("IPAdapterModelLoader", "IPAdapterAdvanced"),
    ),
)
