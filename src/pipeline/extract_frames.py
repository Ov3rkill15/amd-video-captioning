from pathlib import Path

import cv2


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
