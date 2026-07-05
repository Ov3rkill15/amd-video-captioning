import json
from datetime import datetime, timezone
from pathlib import Path


def save_captions_json(output_path: str, video_path: str, frame_captions: list[dict]) -> None:
    payload = {
        "video": video_path,
        "frames": frame_captions,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
