"""Quick sanity check: verify FIREWORKS_API_KEY works and FIREWORKS_VLM_MODEL is accessible.

Usage: python scripts/check_model.py
"""
import os
import sys

import requests
from dotenv import load_dotenv


def main():
    load_dotenv()
    api_key = os.environ.get("FIREWORKS_API_KEY")
    model = os.environ.get("FIREWORKS_VLM_MODEL")

    if not api_key:
        print("FAIL: FIREWORKS_API_KEY is not set in .env")
        sys.exit(1)
    if not model:
        print("FAIL: FIREWORKS_VLM_MODEL is not set in .env")
        sys.exit(1)

    resp = requests.get(
        "https://api.fireworks.ai/inference/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    if resp.status_code != 200:
        print(f"FAIL: could not list models (HTTP {resp.status_code}): {resp.text}")
        sys.exit(1)

    available = {m["id"]: m for m in resp.json().get("data", [])}
    if model not in available:
        print(f"FAIL: '{model}' is not accessible with this API key (serverless).")
        print("Available serverless models:")
        for model_id, info in available.items():
            print(f"  - {model_id} (vision: {info.get('supports_image_input')})")
        sys.exit(1)

    info = available[model]
    if not info.get("supports_image_input"):
        print(f"WARN: '{model}' is accessible but does not support image input (not a VLM).")
        sys.exit(1)

    print(f"OK: '{model}' is accessible and supports image input.")


if __name__ == "__main__":
    main()
