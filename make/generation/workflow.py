"""ComfyUI workflows versioned in the repo as data (API-format JSON).

Workflows are plain ComfyUI API graphs, loadable in ComfyUI as-is. The
pipeline finds the nodes it fills in by their `_meta.title`:

    input:prompt        CLIPTextEncode.text
    input:negative      CLIPTextEncode.text
    input:depth         LoadImage.image
    input:normal        LoadImage.image
    input:reference:N   LoadImage.image (N = 0, 1, ...)
    input:size          EmptyLatentImage.width/height
    input:seed          KSampler.seed
    output:image        SaveImage
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

WORKFLOWS_DIR = Path(__file__).resolve().parent.parent / "workflows"
DEFAULT_WORKFLOW = "view_sdxl_cn_ipadapter"


class Workflow:
    def __init__(self, name: str, graph: dict):
        self.name = name
        self.graph = graph

    @classmethod
    def load(cls, name: str) -> "Workflow":
        return cls(name, json.loads((WORKFLOWS_DIR / f"{name}.json").read_text(encoding="utf-8")))

    @classmethod
    def load_default(cls) -> "Workflow":
        return cls.load(DEFAULT_WORKFLOW)

    def node_types(self) -> set[str]:
        return {node["class_type"] for node in self.graph.values()}

    def node_id(self, title: str) -> str:
        for node_id, node in self.graph.items():
            if node.get("_meta", {}).get("title") == title:
                return node_id
        raise KeyError(f"workflow {self.name!r} has no node titled {title!r}")

    def has(self, title: str) -> bool:
        return any(n.get("_meta", {}).get("title") == title for n in self.graph.values())

    def reference_slots(self) -> int:
        count = 0
        while self.has(f"input:reference:{count}"):
            count += 1
        return count

    def bind(self, values: dict[str, dict]) -> dict:
        """Copy of the graph with `{title: {input: value}}` applied."""
        graph = copy.deepcopy(self.graph)
        for title, inputs in values.items():
            graph[self.node_id(title)]["inputs"].update(inputs)
        return graph
