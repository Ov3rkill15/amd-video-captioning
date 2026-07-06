"""Local demo UI for the Track 2 video-captioning agent.

Stdlib-only (no new dependencies): serves demo/ui.html plus a tiny JSON API
around the same pipeline functions the submission agent uses.

Run from the repo root:  python demo/app.py   ->  http://localhost:8765
"""

import json
import os
import re
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from agent import resolve_video  # noqa: E402
from src.models.fireworks_client import FireworksVLMClient  # noqa: E402
from src.pipeline.extract_frames import sample_frames_b64  # noqa: E402
from src.pipeline.style_caption import generate_styled_captions  # noqa: E402

UI_PATH = Path(__file__).resolve().parent / "ui.html"
PORT = 8765
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter console for demo recordings
        pass

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict):
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self._send(200, UI_PATH.read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/info":
            client = FireworksVLMClient()
            self._json(200, {"model": client.model})
        elif url.path == "/media":
            self._serve_media(url)
        else:
            self._send(404, b"not found", "text/plain")

    def _serve_media(self, url):
        src = parse_qs(url.query).get("src", [""])[0]
        path = Path(src)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
            self._send(404, b"video not found", "text/plain")
            return
        data = path.read_bytes()
        range_header = self.headers.get("Range")
        if range_header:
            m = re.match(r"bytes=(\d+)-(\d*)", range_header)
            if m:
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else len(data) - 1
                chunk = data[start : end + 1]
                self._send(
                    206,
                    chunk,
                    "video/mp4",
                    {
                        "Content-Range": f"bytes {start}-{end}/{len(data)}",
                        "Accept-Ranges": "bytes",
                    },
                )
                return
        self._send(200, data, "video/mp4", {"Accept-Ranges": "bytes"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/generate":
            self._send(404, b"not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length))
            video = req["video"]
            styles = req.get("styles") or ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]

            client = FireworksVLMClient()
            with tempfile.TemporaryDirectory() as work_dir:
                video_path = resolve_video(video, work_dir, "ui")
                frames = sample_frames_b64(video_path, num_frames=8)
                captions = generate_styled_captions(client, frames, styles)
            self._json(200, {"captions": captions, "frames": frames, "model": client.model})
        except FileNotFoundError as e:
            self._json(400, {"error": str(e)})
        except Exception as e:
            self._json(500, {"error": f"{type(e).__name__}: {e}"})


def main():
    os.chdir(REPO_ROOT)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Demo UI running at http://localhost:{PORT}  (Ctrl+C to stop)")
    server.serve_forever()


if __name__ == "__main__":
    main()
