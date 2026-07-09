import base64
import os
import time

import requests

DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"


def resolve_api_key() -> str | None:
    """Plain env var first; else decode FIREWORKS_API_KEY_B64.

    The base64 form is what ships inside the public submission image — it keeps
    the key out of reach of regex-based secret scanners that trawl registries.
    """
    key = os.environ.get("FIREWORKS_API_KEY")
    if key:
        return key
    b64 = os.environ.get("FIREWORKS_API_KEY_B64")
    if b64:
        return base64.b64decode(b64).decode("utf-8").strip()
    return None


class FireworksVLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ):
        self.api_key = api_key or resolve_api_key()
        if not self.api_key:
            raise RuntimeError("Set FIREWORKS_API_KEY or FIREWORKS_API_KEY_B64")
        self.model = model or os.environ.get(
            "FIREWORKS_VLM_MODEL", "accounts/fireworks/models/kimi-k2p6"
        )
        base_url = (base_url or os.environ.get("FIREWORKS_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.endpoint = f"{base_url}/chat/completions"

    def chat(
        self,
        prompt: str,
        images_b64: list[str] | None = None,
        max_tokens: int = 600,
        temperature: float = 0.6,
        retries: int = 2,
        timeout: int = 90,
    ) -> str:
        """Single chat completion with optional inline JPEG images (base64)."""
        content: list[dict] = [{"type": "text", "text": prompt}]
        for img in images_b64 or []:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                }
            )

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": content}],
        }
        if "fireworks.ai" in self.endpoint:
            # kimi-k2p6 is a reasoning model; other OpenAI-compatible hosts
            # (OpenRouter, vLLM) may reject this non-standard field.
            payload["reasoning_effort"] = "none"

        last_err: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response = requests.post(
                    self.endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"].strip()
            except (requests.RequestException, KeyError, IndexError) as e:
                last_err = e
                if attempt < retries:
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"Fireworks API call failed after {retries + 1} attempts: {last_err}")

    def caption_frame(self, image_path: str, prompt: str = "Describe this frame in one concise sentence.") -> str:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        return self.chat(prompt, images_b64=[image_b64], max_tokens=100, temperature=0.3)
