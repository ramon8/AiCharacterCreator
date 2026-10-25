"""Check that a ComfyUI server can run our workflows, naming whatever is missing."""

from __future__ import annotations

from urllib.parse import quote

from make import settings
from make.comfyui import http, manifest
from make.generation.workflow import Workflow


class ComfyUIUnavailable(Exception):
    pass


def probe(url: str, timeout: float = 5) -> None:
    """Raise `http.ServerUnreachable` / `http.HttpError` unless a ComfyUI server answers at `url`."""
    http.get_json(f"{url}/system_stats", timeout=timeout)


def check(url: str | None = None, workflows: list[Workflow] | None = None) -> None:
    """Raise `ComfyUIUnavailable` unless the server can run `workflows` (default: all
    the workflows in the repo)."""
    url = url or settings.comfyui_url()
    workflows = workflows or Workflow.load_all()
    try:
        probe(url)
    except (http.ServerUnreachable, http.HttpError) as error:
        raise ComfyUIUnavailable(
            f"ComfyUI server not responding at {url} ({error}). Start it with `uv run make comfyui start`."
        ) from error

    problems: list[str] = []
    provided_by = {t: node.name for node in manifest.CUSTOM_NODES for t in node.node_types}
    for node_type in sorted(set().union(*(w.node_types() for w in workflows))):
        if not http.get_json(f"{url}/object_info/{quote(node_type)}"):
            source = provided_by.get(node_type)
            hint = f" (custom node {source} not installed)" if source else ""
            problems.append(f"missing node type {node_type}{hint}")

    listings: dict[str, list[str]] = {}
    for model in manifest.MODELS:
        if model.folder not in listings:
            try:
                listings[model.folder] = http.get_json(f"{url}/models/{quote(model.folder)}")
            except http.HttpError:
                listings[model.folder] = []
        if model.filename not in listings[model.folder]:
            problems.append(f"missing model {model.folder}/{model.filename} (download from {model.url})")

    if problems:
        raise ComfyUIUnavailable(
            f"ComfyUI at {url} cannot run workflow(s) {', '.join(w.name for w in workflows)}:\n  - "
            + "\n  - ".join(problems)
            + "\nRun `uv run make comfyui install` to fix."
        )
