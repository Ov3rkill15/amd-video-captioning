# AMD Developer Hackathon: ACT II — Video Captioning Agent

Track: **Video Captioning (Track 2)**
Hackathon: [lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii](https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii)
Timeline: July 6–11, 2026

## What it does
An agent that "watches" a video clip (evenly sampled frames via OpenCV) and uses a
Vision-Language Model on the Fireworks AI API to write captions in the four styles
required by the Track 2 spec: `formal`, `sarcastic`, `humorous_tech`, `humorous_non_tech`.

Two modes:
- **`agent.py` — submission mode (official Track 2 harness contract):** reads
  `/input/tasks.json` (a list of `{task_id, video_url, styles}`), downloads each video,
  samples 8 evenly spaced frames (downscaled to ≤768px), makes ONE multimodal VLM call
  that returns all styles at once (JSON), writes `/output/results.json`, exits 0.
  Tasks run in parallel (4 workers); the 3 official example UHD clips finish in ~25
  seconds — far under the 10-minute budget for the ~12-clip hidden set.
  Defensive by design: API retries, per-style fallback calls if the JSON response
  fails to parse, and generic style-matched fallback captions if a video fails
  entirely (a missing style scores zero).
- **`main.py` — demo/exploration mode (CLI):** timestamped per-frame captions from a
  local video file (`--input video.mp4`), useful for visual demos and debugging.

## Stack
- Python 3.11
- OpenCV (frame sampling), ffmpeg-python (optional audio extraction)
- Fireworks AI API (hosted VLM, default model: `kimi-k2p6` — serverless, vision-capable)
- AMD Developer Cloud / ROCm / MI300X (optional local-model experiments — credits not active as of Jul 5)

## Why not Gemma (for now)
This track was originally attractive partly because of the Google DeepMind
"base use case for Gemma" bonus prize, via `accounts/fireworks/models/gemma-4-26b-a4b-it`
(multimodal, 262k context). Tested on 2026-07-05: on our Fireworks account this model
is **only available via Dedicated Deployment** (hourly GPU rental, not serverless
pay-per-token) — calls to the regular chat completions endpoint always 404.
The pipeline's default VLM is therefore `kimi-k2p6` (serverless, vision-capable,
verified working). A Gemma deployment remains an optional stretch goal if time/budget allows.

## Access status (updated 2026-07-07)
- AMD Developer Cloud account: created, **GPU credits not yet active**
- Fireworks AI API key: **active and verified**
- End-to-end pipeline **tested and working** locally and inside Docker against the
  3 official example clips (see `demo/samples/results_track2_examples.json`)
- Since captioning inference runs on the Fireworks API (not local inference), the
  agent can be fully developed and tested without waiting for AMD credits.
  AMD Developer Cloud is only needed for optional MI300X/ROCm experiments later.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in FIREWORKS_API_KEY
```

## Sample demo videos

`.mp4` files are not committed (see `.gitignore`). The per-frame demo mode uses
Blender Foundation open-movie clips (CC-BY 3.0); download them with:

```bash
curl -L -o demo/samples/bbb_10s.mp4 "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_1MB.mp4"
curl -L -o demo/samples/sintel_10s.mp4 "https://test-videos.co.uk/vids/sintel/mp4/h264/360/Sintel_360_10s_1MB.mp4"
```

Pipeline outputs for both clips live in `demo/samples/output_bbb.json` and
`demo/samples/output_sintel.json` (frames in `frames_bbb/`, `frames_sintel/`).

## Running the submission agent (Track 2 harness contract)

Locally (no Docker), using env overrides:

```bash
INPUT_PATH=demo/samples/tasks_track2_examples.json OUTPUT_PATH=/tmp/results.json python agent.py
```

Via Docker — exactly as the judging harness runs it (the image carries its own `.env`
because Track 2 injects **no** API key; use a spend-limited Fireworks key):

```bash
docker build -t amd-video-captioning .
docker run --rm -v "$(pwd)/input:/input:ro" -v "$(pwd)/output:/output" amd-video-captioning
```

Results for the 3 official example clips are checked in at
`demo/samples/results_track2_examples.json`
(input: `demo/samples/tasks_track2_examples.json`). Exit code 0, ~25 seconds total.

The per-frame demo CLI still works in Docker by overriding the entrypoint:

```bash
docker run --rm --entrypoint python -v "$(pwd)/demo:/app/demo" amd-video-captioning main.py --input demo/samples/bbb_10s.mp4
```

> On Windows Git Bash, if volume mounts don't show up on the host, prefix `docker run`
> with `MSYS_NO_PATHCONV=1` — Git Bash tends to rewrite container paths
> (`/input`, `/output`) into Windows paths.

## Demo UI (local web app)

A stdlib-only local web app for demo recordings: pick a video (path or URL),
hit Generate, and watch the agent sample frames and write all four styled
captions live, next to the video player.

```bash
python demo/app.py   # then open http://localhost:8765
```

No extra dependencies; it reuses the exact same pipeline functions as `agent.py`.

## Running the per-frame demo pipeline

```bash
python main.py --input demo/samples/bbb_10s.mp4 --output demo/samples/output_bbb.json --frames-dir demo/samples/frames_bbb --fps 0.5
```

Optionally extract the audio track too:

```bash
python main.py --input demo/samples/bbb_10s.mp4 --extract-audio
```

## Verifying Fireworks connectivity & model access

Before a full run, verify the API key is valid and the VLM configured in `.env`
is actually accessible (serverless, not dedicated-deployment-only):

```bash
python scripts/check_model.py
```

`OK: '<model>' is accessible and supports image input.` means you're ready.
On `FAIL`, the script lists the serverless models actually accessible to your key.

## Project structure

```
src/
  pipeline/       # extract_frames.py, caption.py, style_caption.py
  models/         # fireworks_client.py
  utils/          # io_utils.py
demo/
  samples/        # sample videos, frame output, daily JSON results
  script.md       # demo recording script
agent.py          # submission ENTRYPOINT — Track 2 harness contract (/input → /output)
main.py           # per-frame demo CLI
Dockerfile
```

## Progress log
<!-- updated daily as evidence of progress -->
- **2026-07-05**: Initial scaffolding — pipeline structure, Fireworks client wrapper, Dockerfile, README.
- **2026-07-05**: Switched default VLM to `gemma-4-26b-a4b-it` (to qualify for the Gemma bonus prize). Fireworks API key active; AMD Developer Cloud credits still pending.
- **2026-07-05**: End-to-end test with a synthetic video: `gemma-4-26b-a4b-it` turned out to 404 (Dedicated-Deployment-only on our account, not serverless). Switched default VLM to `kimi-k2p6` (serverless, vision-capable). Also fixed a bug where kimi-k2 responses contained raw reasoning traces without `reasoning_effort: "none"` — added in `fireworks_client.py`. Pipeline now runs cleanly end to end.
- **2026-07-05**: Docker image built & tested (`docker build` + `docker run` with volume mounts) — captions identical to the local run. Added `scripts/check_model.py` for quick API-key + model-access verification without processing a full video.
- **2026-07-06** (official kickoff): Re-verified API access on day one (`check_model.py` → OK). Replaced synthetic videos with 2 real CC-BY clips (Big Buck Bunny 720p + Sintel 360p, Blender Foundation) — pipeline runs cleanly on both with scene-specific, varied captions (see `output_bbb.json`, `output_sintel.json`). Drafted the full `demo/script.md` (recording flow + judge talking points + license notes).
- **2026-07-07**: Official submission guide released — the Track 2 contract is harness-based: read `/input/tasks.json` (video_url + 4 styles), write `/output/results.json` (one caption per style per video). Major refactor: added `agent.py` (submission entrypoint; video download, 4 parallel workers, layered fallbacks), `src/pipeline/style_caption.py` (one multimodal call → 4 styled captions, robust JSON parsing), `sample_frames_b64()` (8 evenly spaced in-memory frames, ≤768px), upgraded `fireworks_client.py` (multi-image + retries + `FIREWORKS_BASE_URL`). Tested locally and in Docker against the 3 official example clips: exit 0, ~25 seconds, accurate captions with on-target styles (see `results_track2_examples.json`). Image is linux/amd64, 1.2GB. All repo docs rewritten in English.
- **2026-07-08**: Submission image published to `ghcr.io/ov3rkill15/amd-video-captioning:latest` (public, anonymous pull verified, linux/amd64 manifest, 1.21GB compressed). Harness contract re-tested against the published image with the baked-in spend-limited key: 3 official clips, exit 0, ~20 seconds. Cover image and 7-page slide deck added under `demo/submission/`.
- **2026-07-09**: Measurement day. Extended the eval harness (`scripts/eval_captions.py`) with model A/B support: generation can run on any OpenAI-compatible host while the LLM judge stays pinned to the default Fireworks client so scores remain comparable. Results over 6 diverse clips x 4 styles: shipped baseline (kimi-k2p6 + tone exemplars) 0.81; Gemma 4 26B A4B via OpenRouter 0.71 one-stage / 0.77 two-stage (style ties at 0.87 but visual accuracy collapses on two clips); best-of-3 sampling 0.79; one-round self-critique 0.80; two-stage 0.78; 16 frames 0.70 (more frames hurt accuracy). The shipped config won every comparison, so the published image stays as-is. Evidence: `demo/samples/eval_sweep_kimi_2026-07-09.json`, `demo/samples/eval_bestof3_critique_2026-07-09.json`.
- **2026-07-10**: Demo video produced programmatically (ffmpeg + PIL): title/architecture/eval cards plus the 3 official clips with the pipeline's real captions burned in as rotating style banners, then an ElevenLabs voiceover muxed at exact segment timestamps (script and mux command in `demo/narration-voiceover.md`).
