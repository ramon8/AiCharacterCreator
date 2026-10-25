"""In-process stand-in for the ComfyUI HTTP API.

Implements just the endpoints the pipeline talks to, so the real client can
be exercised without a GPU or a ComfyUI install.
"""

from __future__ import annotations

import io
import json
import re
import threading
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from PIL import Image


@dataclass
class FakeComfyUI:
    node_types: set[str] = field(default_factory=set)
    models: dict[str, list[str]] = field(default_factory=dict)
    output_color: tuple[int, int, int] = (200, 100, 50)
    uploads: dict[str, bytes] = field(default_factory=dict)
    prompts: list[dict] = field(default_factory=list)
    execution_error: str | None = None
    frees: list[dict] = field(default_factory=list)
    url: str = ""

    def __post_init__(self) -> None:
        self._history: dict[str, dict] = {}
        self._outputs: dict[str, bytes] = {}
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep test output quiet
                pass

            def _send(self, status: int, body: bytes, content_type="application/json"):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _json(self, data, status=200):
                self._send(status, json.dumps(data).encode())

            def do_GET(self):
                parsed = urlparse(self.path)
                path = parsed.path
                if path == "/system_stats":
                    return self._json({"system": {}, "devices": []})
                if m := re.fullmatch(r"/object_info/(.+)", path):
                    name = m.group(1)
                    return self._json({name: {}} if name in fake.node_types else {})
                if m := re.fullmatch(r"/models/(.+)", path):
                    folder = m.group(1)
                    if folder not in fake.models:
                        return self._send(404, b"")
                    return self._json(fake.models[folder])
                if m := re.fullmatch(r"/history/(.+)", path):
                    pid = m.group(1)
                    return self._json({pid: fake._history[pid]} if pid in fake._history else {})
                if path == "/view":
                    name = parse_qs(parsed.query)["filename"][0]
                    return self._send(200, fake._outputs[name], "image/png")
                self._send(404, b"")

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                if self.path == "/upload/image":
                    name, data = _parse_upload(self.headers["Content-Type"], body)
                    fake.uploads[name] = data
                    return self._json({"name": name, "subfolder": "", "type": "input"})
                if self.path == "/free":
                    fake.frees.append(json.loads(body))
                    return self._send(200, b"")
                if self.path == "/prompt":
                    payload = json.loads(body)
                    return self._json(*fake._run(payload["prompt"]))
                self._send(404, b"")

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def _run(self, graph: dict) -> tuple[dict, int]:
        missing = sorted({n["class_type"] for n in graph.values()} - self.node_types)
        if missing:
            return {"error": {"message": f"missing nodes {missing}"}}, 400
        self.prompts.append(graph)
        pid = str(uuid.uuid4())
        if self.execution_error:
            self._history[pid] = {
                "status": {"status_str": "error", "completed": False,
                           "messages": [["execution_error", {"exception_message": self.execution_error}]]},
                "outputs": {},
            }
            return {"prompt_id": pid, "number": len(self.prompts)}, 200
        save_id = next(k for k, n in graph.items() if n["class_type"] == "SaveImage")
        latent = next((n for n in graph.values() if n["class_type"] == "EmptyLatentImage"), None)
        if latent is not None:
            size = (latent["inputs"]["width"], latent["inputs"]["height"])
        else:  # repaint workflows start from the init image
            init = next(n for n in graph.values() if n["_meta"]["title"] == "input:init")["inputs"]["image"]
            size = Image.open(io.BytesIO(self.uploads[init])).size
        image = Image.new("RGB", size, self.output_color)
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        filename = f"{pid}.png"
        self._outputs[filename] = buffer.getvalue()
        self._history[pid] = {
            "status": {"status_str": "success", "completed": True},
            "outputs": {save_id: {"images": [{"filename": filename, "subfolder": "", "type": "output"}]}},
        }
        return {"prompt_id": pid, "number": len(self.prompts)}, 200

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def _parse_upload(content_type: str, body: bytes) -> tuple[str, bytes]:
    boundary = content_type.split("boundary=")[1].encode()
    for part in body.split(b"--" + boundary):
        head, _, data = part.partition(b"\r\n\r\n")
        if b'name="image"' in head:
            name = re.search(rb'filename="([^"]+)"', head).group(1).decode()
            return name, data.removesuffix(b"\r\n")
    raise ValueError("no image part")
