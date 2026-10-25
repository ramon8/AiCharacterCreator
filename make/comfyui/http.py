"""Minimal JSON/multipart helpers over urllib for the ComfyUI HTTP API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid


class ServerUnreachable(Exception):
    pass


class HttpError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


def _open(request: urllib.request.Request, timeout: float) -> bytes:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        raise HttpError(error.code, error.read().decode(errors="replace")) from error
    except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
        raise ServerUnreachable(str(error)) from error


def get_bytes(url: str, timeout: float = 30) -> bytes:
    return _open(urllib.request.Request(url), timeout)


def get_json(url: str, timeout: float = 30):
    return json.loads(get_bytes(url, timeout))


def post_json(url: str, payload: dict, timeout: float = 30):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    body = _open(request, timeout)
    return json.loads(body) if body.strip() else None


def post_file(url: str, field: str, filename: str, data: bytes, fields: dict[str, str], timeout: float = 60):
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        f"Content-Type: image/png\r\n\r\n".encode() + data + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    request = urllib.request.Request(
        url, data=b"".join(parts), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    return json.loads(_open(request, timeout))
