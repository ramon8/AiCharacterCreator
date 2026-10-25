import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from make.comfyui import install

PAYLOAD = bytes(range(256)) * 64


class _RangeServer(BaseHTTPRequestHandler):
    """Serves PAYLOAD with Range support, answering 416 to a range past its end
    (as Hugging Face's CDN does)."""

    def do_GET(self):
        start = 0
        if header := self.headers.get("Range"):
            start = int(header.removeprefix("bytes=").split("-")[0])
            if start >= len(PAYLOAD):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(PAYLOAD)}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
        else:
            self.send_response(200)
        body = PAYLOAD[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def url(monkeypatch):
    monkeypatch.setattr(install, "_hf_token", lambda: None)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RangeServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/model.safetensors"
    server.shutdown()
    server.server_close()


def test_downloads_the_file(url, tmp_path):
    dest = tmp_path / "model.safetensors"
    install.download(url, dest)
    assert dest.read_bytes() == PAYLOAD


def test_resumes_a_partial_download(url, tmp_path):
    dest = tmp_path / "model.safetensors"
    dest.with_name(dest.name + ".part").write_bytes(PAYLOAD[:1000])
    install.download(url, dest)
    assert dest.read_bytes() == PAYLOAD


def test_a_partial_download_that_is_already_complete_is_kept(url, tmp_path):
    dest = tmp_path / "model.safetensors"
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(PAYLOAD)
    install.download(url, dest)
    assert dest.read_bytes() == PAYLOAD
    assert not part.exists()


def test_a_partial_download_longer_than_the_file_is_downloaded_again(url, tmp_path):
    dest = tmp_path / "model.safetensors"
    dest.with_name(dest.name + ".part").write_bytes(PAYLOAD + b"junk")
    install.download(url, dest)
    assert dest.read_bytes() == PAYLOAD
