import base64
import os

import requests

FIREWORKS_ENDPOINT = "https://api.fireworks.ai/inference/v1/chat/completions"


class FireworksVLMClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.environ["FIREWORKS_API_KEY"]
        self.model = model or os.environ.get(
            "FIREWORKS_VLM_MODEL", "accounts/fireworks/models/gemma-4-26b-a4b-it"
        )

    def caption_frame(self, image_path: str, prompt: str = "Describe this frame in one concise sentence.") -> str:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        response = requests.post(
            FIREWORKS_ENDPOINT,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 100,
                "reasoning_effort": "none",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                            },
                        ],
                    }
                ],
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
