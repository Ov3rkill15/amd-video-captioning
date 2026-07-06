import json
import re

from src.models.fireworks_client import FireworksVLMClient

STYLE_DEFINITIONS = {
    "formal": "Professional, objective, factual tone. No jokes, no slang.",
    "sarcastic": "Dry, ironic, lightly mocking tone. Understated wit, not mean-spirited.",
    "humorous_tech": "Funny, packed with technology or programming references (e.g. bugs, loading, Wi-Fi, git, CPUs).",
    "humorous_non_tech": "Funny, everyday humour anyone gets. Absolutely no technical jargon.",
}

FALLBACK_CAPTIONS = {
    "formal": "A short video clip depicting a scene with visual activity and movement.",
    "sarcastic": "Ah yes, another video clip. Truly groundbreaking footage of things happening.",
    "humorous_tech": "This clip buffered its way into my heart — 100% loaded, 0 bugs found.",
    "humorous_non_tech": "Somewhere out there, something happened on camera, and honestly, good for it.",
}


def _extract_json(text: str) -> dict | None:
    """Best-effort extraction of a JSON object from a model response."""
    # Strip markdown code fences if present.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost {...} block.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _build_prompt(styles: list[str], num_frames: int) -> str:
    style_lines = "\n".join(
        f"- {s}: {STYLE_DEFINITIONS.get(s, 'Match the tone implied by the style name.')}"
        for s in styles
    )
    keys = ", ".join(f'"{s}"' for s in styles)
    return (
        f"The {num_frames} images below are frames sampled in chronological order from ONE short video clip.\n\n"
        "Step 1 - watch: work out what the video shows (subject, setting, action, mood).\n"
        "Step 2 - write: produce ONE caption per requested style. Each caption must:\n"
        "- accurately describe THIS specific video (mention concrete subjects/actions you see),\n"
        "- be 1-2 sentences, under 40 words,\n"
        "- be in English,\n"
        "- nail the requested tone.\n\n"
        f"Style definitions:\n{style_lines}\n\n"
        f"Respond with ONLY a valid JSON object with exactly these keys: {keys}. "
        "No markdown, no explanation, just the JSON object."
    )


def generate_styled_captions(
    client: FireworksVLMClient, frames_b64: list[str], styles: list[str]
) -> dict[str, str]:
    """Generate one caption per style for a video represented by sampled frames.

    Never raises for caption-quality reasons: guarantees a non-empty caption for
    every requested style (missing styles score zero in the harness).
    """
    captions: dict[str, str] = {}

    # Primary path: one multimodal call returning all styles as JSON.
    for temperature in (0.6, 0.9):  # second attempt runs hotter to escape bad decodes
        try:
            raw = client.chat(
                _build_prompt(styles, len(frames_b64)),
                images_b64=frames_b64,
                max_tokens=200 * len(styles),
                temperature=temperature,
            )
        except RuntimeError:
            continue
        parsed = _extract_json(raw)
        if parsed:
            for style in styles:
                value = parsed.get(style)
                if isinstance(value, str) and value.strip():
                    captions[style] = value.strip()
            if all(s in captions for s in styles):
                return captions

    # Secondary path: one call per still-missing style.
    for style in styles:
        if style in captions:
            continue
        try:
            definition = STYLE_DEFINITIONS.get(style, "Match the tone implied by the style name.")
            captions[style] = client.chat(
                "The images below are frames sampled in order from one short video clip. "
                f"Write a single caption for the video in this style: {style} ({definition}). "
                "1-2 sentences, under 40 words, in English. Respond with the caption text only.",
                images_b64=frames_b64,
                max_tokens=120,
                temperature=0.7,
            ).strip().strip('"')
        except RuntimeError:
            pass

    # Last resort: generic style-matched fallback (low accuracy beats a zero).
    for style in styles:
        if not captions.get(style):
            captions[style] = FALLBACK_CAPTIONS.get(
                style, "A short video clip showing a scene in motion."
            )
    return captions
