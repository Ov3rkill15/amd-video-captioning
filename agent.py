"""AMD ACT II Track 2 submission entrypoint.

Harness contract:
- read  /input/tasks.json   [{"task_id", "video_url", "styles": [...]}, ...]
- write /output/results.json [{"task_id", "captions": {style: caption}}, ...]
- exit 0 on success, non-zero on failure; total runtime budget 10 minutes.

INPUT_PATH / OUTPUT_PATH env vars override the paths for local development.
"""

import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv

from src.models.fireworks_client import FireworksVLMClient
from src.pipeline.extract_frames import sample_frames_b64
from src.pipeline.style_caption import (
    FALLBACK_CAPTIONS,
    generate_styled_captions,
    generate_verified_captions,
)

DEFAULT_STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
# "verified" = describe -> verify -> per-style writer; "oneshot" = single VLM call.
CAPTION_MODE = os.environ.get("CAPTION_MODE", "verified").strip().lower()
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "5" if CAPTION_MODE == "verified" else "8"))
MAX_WORKERS = 4
DOWNLOAD_TIMEOUT = 180


def resolve_video(url: str, dest_dir: str, task_id: str) -> str:
    """Download an http(s) video URL; anything else is treated as a local path."""
    if not url.lower().startswith(("http://", "https://")):
        if not Path(url).is_file():
            raise FileNotFoundError(f"Local video not found: {url}")
        return url
    path = str(Path(dest_dir) / f"video_{task_id}.mp4")
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return path


def process_task(task: dict, client: FireworksVLMClient, work_dir: str) -> dict:
    task_id = str(task.get("task_id", ""))
    styles = task.get("styles") or DEFAULT_STYLES
    started = time.time()
    try:
        video_path = resolve_video(task["video_url"], work_dir, task_id)
        frames = sample_frames_b64(video_path, num_frames=NUM_FRAMES)
        if CAPTION_MODE == "verified":
            captions = generate_verified_captions(client, frames, styles)
        else:
            captions = generate_styled_captions(client, frames, styles)
        if video_path.startswith(work_dir):  # only delete files we downloaded
            Path(video_path).unlink(missing_ok=True)
        print(f"[task {task_id}] done in {time.time() - started:.1f}s", flush=True)
    except Exception as e:
        # A broken clip must never sink the whole run; emit style-matched fallbacks.
        print(f"[task {task_id}] FAILED ({e}); using fallback captions", flush=True)
        captions = {
            s: FALLBACK_CAPTIONS.get(s, "A short video clip showing a scene in motion.")
            for s in styles
        }
    return {"task_id": task_id, "captions": captions}


def main() -> int:
    load_dotenv()  # local dev + credentials baked into the image

    input_path = os.environ.get("INPUT_PATH", "/input/tasks.json")
    output_path = os.environ.get("OUTPUT_PATH", "/output/results.json")

    try:
        with open(input_path, encoding="utf-8") as f:
            tasks = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"FATAL: cannot read tasks from {input_path}: {e}", file=sys.stderr)
        return 1
    if not isinstance(tasks, list):
        print(f"FATAL: {input_path} must contain a JSON array", file=sys.stderr)
        return 1

    client = FireworksVLMClient()
    results: list[dict] = []
    with tempfile.TemporaryDirectory() as work_dir:
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, max(1, len(tasks)))) as pool:
            futures = {pool.submit(process_task, t, client, work_dir): t for t in tasks}
            for future in as_completed(futures):
                results.append(future.result())

    # Preserve input task order for readability.
    order = {str(t.get("task_id", "")): i for i, t in enumerate(tasks)}
    results.sort(key=lambda r: order.get(r["task_id"], len(order)))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(results)} result(s) to {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
