import base64
from pathlib import Path

import cv2


def sample_frames_b64(video_path: str, num_frames: int = 8, max_side: int = 768) -> list[str]:
    """Sample `num_frames` evenly spaced frames as base64 JPEGs (in memory).

    Frames are downscaled so the longest side is at most `max_side` to keep
    request payloads and vision-token usage small.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # Fallback for streams that don't report frame count: read sequentially.
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise ValueError(f"No frames decoded from: {video_path}")
        step = max(1, len(frames) // num_frames)
        selected = frames[::step][:num_frames]
    else:
        indices = [int(i * (total - 1) / max(1, num_frames - 1)) for i in range(min(num_frames, total))]
        selected = []
        for idx in dict.fromkeys(indices):  # dedupe, keep order
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                selected.append(frame)
        cap.release()
        if not selected:
            raise ValueError(f"No frames decoded from: {video_path}")

    encoded = []
    for frame in selected:
        h, w = frame.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1.0:
            frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            encoded.append(base64.b64encode(buf.tobytes()).decode("utf-8"))
    if not encoded:
        raise ValueError(f"Failed to encode frames from: {video_path}")
    return encoded


def extract_frames(video_path: str, output_dir: str, fps: float = 1.0) -> list[dict]:
    """Extract frames from video at the given sampling rate (frames per second).

    Returns a list of {"frame_index", "timestamp_sec", "path"} for each saved frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sample_every_n = max(1, round(video_fps / fps))

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    frames = []
    frame_index = 0
    saved_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % sample_every_n == 0:
            timestamp_sec = frame_index / video_fps
            frame_path = str(Path(output_dir) / f"frame_{saved_index:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append(
                {
                    "frame_index": saved_index,
                    "timestamp_sec": round(timestamp_sec, 2),
                    "path": frame_path,
                }
            )
            saved_index += 1
        frame_index += 1

    cap.release()
    return frames


def extract_audio(video_path: str, output_path: str) -> str | None:
    """Extract audio track to a .wav file using ffmpeg. Returns None if no audio stream."""
    import ffmpeg

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        (
            ffmpeg.input(video_path)
            .output(output_path, acodec="pcm_s16le", ac=1, ar="16000")
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error:
        return None
    return output_path
