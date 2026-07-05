import argparse

from dotenv import load_dotenv

from src.models.fireworks_client import FireworksVLMClient
from src.pipeline.caption import caption_frames
from src.pipeline.extract_frames import extract_audio, extract_frames
from src.utils.io_utils import save_captions_json


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Video captioning pipeline (AMD ACT II)")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output", default="demo/samples/output.json", help="Path to output JSON")
    parser.add_argument("--frames-dir", default="demo/samples/frames", help="Directory to store extracted frames")
    parser.add_argument("--fps", type=float, default=1.0, help="Frame sampling rate (frames per second)")
    parser.add_argument("--extract-audio", action="store_true", help="Also extract the audio track")
    args = parser.parse_args()

    frames = extract_frames(args.input, args.frames_dir, fps=args.fps)
    print(f"Extracted {len(frames)} frames")

    if args.extract_audio:
        audio_path = extract_audio(args.input, "demo/samples/audio.wav")
        print(f"Audio extracted to: {audio_path}" if audio_path else "No audio stream found")

    client = FireworksVLMClient()
    captioned_frames = caption_frames(frames, client)

    save_captions_json(args.output, args.input, captioned_frames)
    print(f"Saved captions to: {args.output}")


if __name__ == "__main__":
    main()
