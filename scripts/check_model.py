"""Quick sanity check: verify FIREWORKS_API_KEY works and FIREWORKS_VLM_MODEL is accessible.

Usage: python scripts/check_model.py
"""
import base64
import os
import sys

import requests
from dotenv import load_dotenv


def main():
    load_dotenv()
    # Same resolution order as src/models/fireworks_client.py: plain key first,
    # then the base64 form that ships inside the public submission image.
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key and os.environ.get("FIREWORKS_API_KEY_B64"):
        api_key = base64.b64decode(os.environ["FIREWORKS_API_KEY_B64"]).decode("utf-8").strip()
    model = os.environ.get("FIREWORKS_VLM_MODEL")

    if not api_key:
        print("FAIL: neither FIREWORKS_API_KEY nor FIREWORKS_API_KEY_B64 is set in .env")
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
