import json
import re

from src.models.fireworks_client import FireworksVLMClient

STYLE_DEFINITIONS = {
    "formal": "Professional, objective, factual tone. No jokes, no slang.",
    "sarcastic": "Dry, ironic, lightly mocking tone. Understated wit, not mean-spirited.",
    "humorous_tech": "Funny, packed with technology or programming references (e.g. bugs, loading, Wi-Fi, git, CPUs).",
    "humorous_non_tech": "Funny, everyday humour anyone gets. Absolutely no technical jargon.",
}

# Tone exemplars from an unrelated imaginary clip. Teaching voice by example
# lifted judged accuracy 0.64->0.75 and style match 0.82->0.89 in eval
# (scripts/eval_captions.py); exemplars describe a DIFFERENT video so no
# content can leak into the output.
TONE_EXEMPLARS = (
    "Example captions for a DIFFERENT video (a dog catching a frisbee in a park), "
    "showing the expected voice per style:\n"
    '- formal: "A border collie leaps to catch a frisbee in a grassy park while its owner watches nearby."\n'
    '- sarcastic: "Yes, the dog caught the frisbee again. Someone alert the sports networks."\n'
    '- humorous_tech: "Latency: 0ms. This dog\'s frisbee-interception algorithm is fully optimized, no patch needed."\n'
    '- humorous_non_tech: "Somewhere a dog just made catching things look easier than most of us make walking."\n\n'
)

FALLBACK_CAPTIONS = {
    "formal": "A short video clip depicting a scene with visual activity and movement.",
    "sarcastic": "Ah yes, another video clip. Truly groundbreaking footage of things happening.",
    "humorous_tech": "This clip buffered its way into my heart: 100% loaded, 0 bugs found.",
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
        TONE_EXEMPLARS
        + f"The {num_frames} images below are frames sampled in chronological order from ONE short video clip.\n\n"
        "Step 1 - watch: work out what the video shows (subject, setting, action, mood).\n"
        "Step 2 - write: produce ONE caption per requested style. Each caption must:\n"
        "- accurately describe THIS specific video (mention concrete subjects/actions you see),\n"
        "- be 1-2 sentences, under 40 words,\n"
        "- be in English,\n"
        "- use plain punctuation: no em-dashes or en-dashes, prefer commas or periods,\n"
        "- nail the requested tone.\n\n"
        f"Style definitions:\n{style_lines}\n\n"
        f"Respond with ONLY a valid JSON object with exactly these keys: {keys}. "
        "No markdown, no explanation, just the JSON object."
    )


# --- verified-scene mode -----------------------------------------------------
# Three stages: (1) describe the clip from frames, (2) re-check the draft
# against the same frames and strip anything unsupported, (3) write one rich
# caption per style from the verified description only. Longer captions
# (25-60 words) give an LLM judge more matched detail to reward on both
# accuracy and style than the one-liner baseline.

STYLE_WRITER_GUIDES = {
    "formal": (
        "Voice: polished, neutral, factual. Zero humor, zero opinion. "
        'Tone reference (different video): "A cyclist in a yellow jacket crosses a rain-soaked intersection while traffic waits at the light."'
    ),
    "sarcastic": (
        "Voice: dry, deadpan, lightly mocking the ordinariness or irony of what happens, never cruel. "
        'Tone reference (different video): "Truly historic footage of a man successfully opening an umbrella, a feat centuries of engineering made possible."'
    ),
    "humorous_tech": (
        "Voice: comedy built on one sustained software or hardware metaphor (debugging, deploys, APIs, latency, caching, game physics). "
        'Tone reference (different video): "The pigeon retries its landing loop three times before the rooftop finally accepts the connection."'
    ),
    "humorous_non_tech": (
        "Voice: warm observational comedy any audience gets, no technical vocabulary at all. "
        'Tone reference (different video): "She organizes the picnic like a general planning a campaign, if generals worried this much about sandwich symmetry."'
    ),
}


def describe_video(client: FireworksVLMClient, frames_b64: list[str]) -> str:
    """Stages 1+2: dense factual description of the clip, then a verify pass."""
    draft = client.chat(
        f"These {len(frames_b64)} images are frames sampled in chronological order from one short video. "
        "Write 2-4 dense factual sentences describing the video: setting, main subjects, "
        "what they do, how the action or camera changes over time, and clearly readable "
        "large on-screen text if any. Be concrete and specific. Do not speculate about "
        "identity, brand, exact location, or intent unless it is unmistakably visible. "
        "Do not mention frames, images, or that this is an analysis. Description only.",
        images_b64=frames_b64,
        max_tokens=240,
        temperature=0.2,
    ).strip()
    return client.chat(
        f"Draft description of this video:\n{draft}\n\n"
        "Compare the draft against the video frames shown. Keep every sentence that is "
        "accurate and specific. Fix anything wrong, vague, or not actually visible. "
        "Drop guessed brands, place names, identities, and quoted text unless clearly "
        "readable and central. Output only the corrected factual description, nothing else. "
        "Do not mention frames, drafts, or checking.",
        images_b64=frames_b64,
        max_tokens=240,
        temperature=0.1,
    ).strip()


def write_style_caption(
    client: FireworksVLMClient, description: str, style: str, prior: list[str]
) -> str:
    """Stage 3: one 25-60 word caption in the target style, text-only call."""
    guide = STYLE_WRITER_GUIDES.get(
        style, f"Voice: match the tone implied by the style name '{style}'."
    )
    variety = ""
    if prior:
        variety = (
            "\n\nCaptions already written for this clip in other styles, "
            "use a clearly different opening and comedic angle:\n"
            + "\n".join(f"- {c}" for c in prior)
        )
    creative = style != "formal"
    return client.chat(
        f"{guide}\n\n"
        f"What the video shows:\n{description}\n\n"
        "Write ONE caption for this video, 25 to 60 words, in English, as someone who "
        "watched the whole clip. Ground every concrete claim in the description above; "
        "invent nothing beyond it. The tone reference describes a DIFFERENT video: copy "
        "only its voice, never its objects, imagery, or wording. Never mention AI, models, "
        "frames, descriptions, or analysis. Plain punctuation only, no em-dashes. "
        "Return only the caption text."
        f"{variety}",
        max_tokens=160,
        temperature=0.75 if creative else 0.2,
    ).strip().strip('"')


def generate_verified_captions(
    client: FireworksVLMClient, frames_b64: list[str], styles: list[str]
) -> dict[str, str]:
    """Verified-scene pipeline; falls back to the one-shot path on stage failure."""
    try:
        description = describe_video(client, frames_b64)
    except RuntimeError:
        return generate_styled_captions(client, frames_b64, styles)

    captions: dict[str, str] = {}
    prior: list[str] = []
    for style in styles:
        try:
            caption = write_style_caption(client, description, style, prior)
        except RuntimeError:
            caption = ""
        if caption:
            captions[style] = caption
            prior.append(caption)
        else:
            captions[style] = FALLBACK_CAPTIONS.get(
                style, "A short video clip showing a scene in motion."
            )
    return captions


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
                "1-2 sentences, under 40 words, in English, plain punctuation (no em-dashes). "
                "Respond with the caption text only.",
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
