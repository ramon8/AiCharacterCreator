import pytest

from make.comfyui import manifest
from make.generation.workflow import Workflow

from fake_comfyui import FakeComfyUI


def complete_install() -> tuple[set[str], dict[str, list[str]]]:
    """Node types and model listings of a ComfyUI with everything installed."""
    nodes = Workflow.load_default().node_types()
    models: dict[str, list[str]] = {}
    for model in manifest.MODELS:
        models.setdefault(model.folder, []).append(model.filename)
    return set(nodes), models


@pytest.fixture
def comfy():
    nodes, models = complete_install()
    server = FakeComfyUI(node_types=nodes, models=models)
    yield server
    server.stop()
