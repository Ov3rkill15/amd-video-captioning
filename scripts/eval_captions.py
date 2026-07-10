"""A/B evaluation harness for caption quality, mirroring the judges' rubric.

Compares pipeline variants on a diverse clip set; an LLM judge scores each
caption on accuracy (0-1) and style match (0-1), like the official LLM-Judge.

Usage: python scripts/eval_captions.py [--variants baseline,fewshot,...]
Results: eval_results.json + a compact table on stdout.
"""

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

import requests  # noqa: E402

from src.models.fireworks_client import FireworksVLMClient  # noqa: E402
from src.pipeline.extract_frames import sample_frames_b64  # noqa: E402
from src.pipeline.style_caption import (  # noqa: E402
    STYLE_DEFINITIONS,
    TONE_EXEMPLARS,
    _build_prompt,
    _extract_json,
    generate_styled_captions,
    generate_verified_captions,
)

STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]

CLIPS = {
    "traffic": "https://storage.googleapis.com/amd-hackathon-clips/1860079-uhd_2560_1440_25fps.mp4",
    "kitten": "https://storage.googleapis.com/amd-hackathon-clips/13825391-uhd_3840_2160_30fps.mp4",
    "office": "https://storage.googleapis.com/amd-hackathon-clips/3044693-uhd_3840_2160_24fps.mp4",
    "bbb": "demo/samples/bbb_10s.mp4",
    "sintel": "demo/samples/sintel_10s.mp4",
    "beach": "D:/PRIBADI/videotest.mp4",
}

CACHE = Path(tempfile.gettempdir()) / "amd_eval_clips"

# NOTE: the fewshot variant won the 2026-07-07 eval and was promoted into
# _build_prompt (TONE_EXEMPLARS), so "baseline" now includes it and the
# "fewshot" variant is kept only for regression comparison.
FEWSHOT = TONE_EXEMPLARS


def get_video(name: str, src: str) -> str:
    if not src.startswith("http"):
        p = src if Path(src).is_absolute() else str(REPO_ROOT / src)
        return p
    CACHE.mkdir(exist_ok=True)
    dest = CACHE / f"{name}.mp4"
    if not dest.exists():
        with requests.get(src, stream=True, timeout=180) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    return str(dest)


# ---------------- variants ----------------

def v_baseline(client, path):
    return generate_styled_captions(client, sample_frames_b64(path, 8), STYLES)


def v_frames16(client, path):
    return generate_styled_captions(client, sample_frames_b64(path, 16), STYLES)


def v_fewshot(client, path):
    frames = sample_frames_b64(path, 8)
    # _build_prompt already includes TONE_EXEMPLARS since the promotion.
    raw = client.chat(_build_prompt(STYLES, len(frames)), images_b64=frames, max_tokens=800, temperature=0.6)
    parsed = _extract_json(raw) or {}
    return {s: (parsed.get(s) or "").strip() for s in STYLES}


def v_twostage(client, path):
    frames = sample_frames_b64(path, 8)
    desc = client.chat(
        f"The {len(frames)} images are frames sampled in order from one short video. "
        "Describe the video in detail: subjects, actions, setting, camera movement, "
        "mood, notable details. 120 words max.",
        images_b64=frames,
        max_tokens=300,
        temperature=0.3,
    )
    style_lines = "\n".join(f"- {s}: {STYLE_DEFINITIONS[s]}" for s in STYLES)
    keys = ", ".join(f'"{s}"' for s in STYLES)
    raw = client.chat(
        f"Video description:\n{desc}\n\n"
        "Write one caption per style for this video. Each: 1-2 sentences, under 40 words, "
        f"English, plain punctuation, accurate to the description, tone must land.\n"
        f"Styles:\n{style_lines}\n\n"
        f"Respond with ONLY a JSON object with keys: {keys}.",
        max_tokens=800,
        temperature=0.7,
    )
    parsed = _extract_json(raw) or {}
    return {s: (parsed.get(s) or "").strip() for s in STYLES}


def v_combo(client, path):
    """Two-stage with few-shot tone exemplars in the writing stage."""
    frames = sample_frames_b64(path, 8)
    desc = client.chat(
        f"The {len(frames)} images are frames sampled in order from one short video. "
        "Describe the video in detail: subjects, actions, setting, camera movement, "
        "mood, notable details. 120 words max.",
        images_b64=frames,
        max_tokens=300,
        temperature=0.3,
    )
    style_lines = "\n".join(f"- {s}: {STYLE_DEFINITIONS[s]}" for s in STYLES)
    keys = ", ".join(f'"{s}"' for s in STYLES)
    raw = client.chat(
        FEWSHOT
        + f"Video description:\n{desc}\n\n"
        "Write one caption per style for this video. Each: 1-2 sentences, under 40 words, "
        "English, plain punctuation, accurate to the description, tone must land.\n"
        f"Styles:\n{style_lines}\n\n"
        f"Respond with ONLY a JSON object with keys: {keys}.",
        max_tokens=800,
        temperature=0.7,
    )
    parsed = _extract_json(raw) or {}
    return {s: (parsed.get(s) or "").strip() for s in STYLES}


def v_bestof3(client, path):
    """Test-time compute: 3 candidate sets, an internal judge picks per style."""
    frames = sample_frames_b64(path, 8)
    prompt = _build_prompt(STYLES, len(frames))
    candidates = []
    for _ in range(3):
        raw = client.chat(prompt, images_b64=frames, max_tokens=800, temperature=0.8)
        parsed = _extract_json(raw) or {}
        candidates.append({s: (parsed.get(s) or "").strip() for s in STYLES})

    style_lines = "\n".join(f"- {s}: {STYLE_DEFINITIONS[s]}" for s in STYLES)
    cand_lines = "\n".join(
        f"{s}:\n" + "\n".join(f"  {i + 1}. \"{c[s]}\"" for i, c in enumerate(candidates))
        for s in STYLES
    )
    raw = client.chat(
        "The images are frames from one video. For each style below, pick the "
        "candidate caption that is most accurate to the video AND lands the tone "
        "best. Prefer specific, visually grounded captions over generic ones.\n\n"
        f"Style definitions:\n{style_lines}\n\nCandidates:\n{cand_lines}\n\n"
        'Respond ONLY with JSON mapping style to candidate number, e.g. {"formal": 2, ...}',
        images_b64=frames,
        max_tokens=200,
        temperature=0.0,
    )
    picks = _extract_json(raw) or {}
    out = {}
    for s in STYLES:
        try:
            idx = int(picks.get(s, 1)) - 1
        except (TypeError, ValueError):
            idx = 0
        out[s] = candidates[idx if 0 <= idx < len(candidates) else 0][s]
    return out


def v_critique(client, path):
    """Baseline captions, then one revision pass against the frames."""
    frames = sample_frames_b64(path, 8)
    captions = generate_styled_captions(client, frames, STYLES)
    style_lines = "\n".join(f"- {s}: {STYLE_DEFINITIONS[s]}" for s in STYLES)
    caps = "\n".join(f'- {s}: "{captions.get(s, "")}"' for s in STYLES)
    keys = ", ".join(f'"{s}"' for s in STYLES)
    raw = client.chat(
        "The images are frames from one video. Verify each caption below against "
        "the frames. Rewrite any caption that claims something not visible in the "
        "video, describes the wrong subject or setting, or is generic enough to fit "
        "any video; make it specific to what the frames show. Keep the tone of its "
        "style, 1-2 sentences, under 40 words, plain punctuation. If a caption is "
        "already accurate and on-tone, return it unchanged.\n\n"
        f"Style definitions:\n{style_lines}\n\nCaptions:\n{caps}\n\n"
        f"Respond with ONLY a JSON object with keys: {keys}.",
        images_b64=frames,
        max_tokens=800,
        temperature=0.4,
    )
    parsed = _extract_json(raw) or {}
    return {s: ((parsed.get(s) or "").strip() or captions.get(s, "")) for s in STYLES}


def v_verified(client, path):
    """Describe -> verify -> per-style 25-60 word captions, 5 frames."""
    return generate_verified_captions(client, sample_frames_b64(path, 5), STYLES)


VARIANTS = {
    "baseline": v_baseline,
    "verified": v_verified,
    "fewshot": v_fewshot,
    "twostage": v_twostage,
    "frames16": v_frames16,
    "combo": v_combo,
    "bestof3": v_bestof3,
    "critique": v_critique,
}


# ---------------- model A/B ----------------

def default_client():
    return FireworksVLMClient()


def gemma_client():
    """Gemma on an OpenAI-compatible host (OpenRouter by default)."""
    key = os.environ.get("GEMMA_API_KEY")
    if not key:
        raise RuntimeError("Set GEMMA_API_KEY (e.g. an OpenRouter key) to run gemma variants")
    return FireworksVLMClient(
        api_key=key,
        model=os.environ.get("GEMMA_MODEL", "google/gemma-4-26b-a4b-it"),
        base_url=os.environ.get("GEMMA_BASE_URL", "https://openrouter.ai/api/v1"),
    )


# Same prompts, different VLM. Excluded from the default run because they
# need GEMMA_API_KEY; request explicitly: --variants baseline,gemma
MODEL_VARIANTS = {
    "gemma": (gemma_client, v_baseline),
    "gemma_twostage": (gemma_client, v_twostage),
}


# ---------------- judge ----------------

def judge(client, path, captions) -> dict:
    frames = sample_frames_b64(path, 4)
    style_lines = "\n".join(f"- {s}: {STYLE_DEFINITIONS[s]}" for s in STYLES)
    caps = "\n".join(f'- {s}: "{captions.get(s, "")}"' for s in STYLES)
    raw = client.chat(
        "You are a strict caption judge. The images are frames from one video. "
        "Score each caption on two dimensions, each 0.0-1.0:\n"
        "accuracy: does it faithfully reflect THIS video's visual content? "
        "(generic captions that fit any video score <=0.3)\n"
        "style: does the tone match the style definition?\n\n"
        f"Style definitions:\n{style_lines}\n\nCaptions:\n{caps}\n\n"
        'Respond ONLY with JSON: {"<style>": {"accuracy": x, "style": y}, ...}',
        images_b64=frames,
        max_tokens=400,
        temperature=0.0,
    )
    return _extract_json(raw) or {}


# ---------------- run ----------------

def run_cell(variant, factory, fn, clip, path):
    t0 = time.time()
    try:
        captions = fn(factory(), path)
        # The judge always runs on the default Fireworks client so scores
        # stay comparable across model variants.
        scores = judge(default_client(), path, captions)
        return variant, clip, {"captions": captions, "scores": scores, "seconds": round(time.time() - t0, 1)}
    except Exception as e:
        return variant, clip, {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS))
    args = ap.parse_args()
    all_variants = {name: (default_client, fn) for name, fn in VARIANTS.items()}
    all_variants.update(MODEL_VARIANTS)
    chosen = {k: all_variants[k] for k in args.variants.split(",")}

    paths = {name: get_video(name, src) for name, src in CLIPS.items()}
    print(f"clips ready: {list(paths)}", flush=True)

    results: dict = {v: {} for v in chosen}
    jobs = [(v, factory, fn, c, p) for v, (factory, fn) in chosen.items() for c, p in paths.items()]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for v, c, cell in pool.map(lambda j: run_cell(*j), jobs):
            results[v][c] = cell
            print(f"  done: {v} x {c}", flush=True)

    out = REPO_ROOT / "eval_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'variant':<10} {'accuracy':>8} {'style':>6} {'total':>6}  (mean over clips x styles)")
    for v, clips in results.items():
        acc, sty = [], []
        for cell in clips.values():
            for s in STYLES:
                sc = cell.get("scores", {}).get(s)
                if isinstance(sc, dict):
                    acc.append(float(sc.get("accuracy", 0)))
                    sty.append(float(sc.get("style", 0)))
        if acc:
            a, st = statistics.mean(acc), statistics.mean(sty)
            print(f"{v:<10} {a:>8.3f} {st:>6.3f} {(a + st) / 2:>6.3f}")
        else:
            print(f"{v:<10} {'ERROR':>8}")
    print(f"\ndetails: {out}")


if __name__ == "__main__":
    main()
