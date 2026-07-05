from tqdm import tqdm

from src.models.fireworks_client import FireworksVLMClient


def caption_frames(frames: list[dict], client: FireworksVLMClient) -> list[dict]:
    """Attach a `caption` field to each frame dict by calling the VLM."""
    captioned = []
    for frame in tqdm(frames, desc="Captioning frames"):
        caption = client.caption_frame(frame["path"])
        captioned.append({**frame, "caption": caption})
    return captioned
