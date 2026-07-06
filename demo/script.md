# Demo Script — AMD ACT II Video Captioning (Track 2)

> Updated Jul 7: demo flow reworked to follow the official submission guide
> (harness-based: `/input/tasks.json` → `/output/results.json`, 4 styles per video).
> Finalize on Jul 10 (day before the deadline).

## Pre-recording checklist
- [ ] `.env` contains a valid `FIREWORKS_API_KEY` (use a spend-limited key — it gets baked into the public image)
- [ ] `python scripts/check_model.py` → must print `OK` (proof the API + model are ready)
- [ ] Docker image built: `docker build -t amd-video-captioning .`
- [ ] `input/` folder contains `tasks.json` with the 3 official example clips (copy from `demo/samples/tasks_track2_examples.json`)
- [ ] Clean terminal, font large enough for recording

## Demo flow (recording order)
1. **Intro (20s):** Video Captioning track — an agent that watches a video clip and
   writes captions in 4 styles (formal, sarcastic, humorous_tech, humorous_non_tech).
   Stack: Python + OpenCV + Fireworks VLM `kimi-k2p6` (serverless). Fully containerized,
   MIT licensed, compliant with the official harness contract.
2. **Show the input (15s):** open `input/tasks.json` — exactly the judges' format
   (`task_id`, `video_url`, `styles`).
3. **Run the container (45s):** exactly as the judging harness does:
   ```bash
   docker run --rm -v "$(pwd)/input:/input:ro" -v "$(pwd)/output:/output" amd-video-captioning
   ```
   Highlight: 3 UHD clips processed in parallel, done in ~25 seconds (the judges'
   budget is 10 minutes for ~12 clips), exit code 0.
4. **Show the output (60s):** open `output/results.json` side by side with clip
   thumbnails. Read out the 1-2 best captions per clip — show that captions are
   accurate to the content (kitten, traffic, office) AND nail the tone
   (e.g. the kitten's humorous_tech: "Kitten.exe has finished loading...").
5. **Architecture in brief (30s):** flow diagram: download → sample 8 evenly spaced
   frames (OpenCV, downscaled to 768px) → ONE multimodal call → JSON with 4 captions.
   Mention the layered fallbacks: API retries → per-style calls → generic captions
   (no style is ever missing = never a zero for a missing style).
6. **Generalisation (20s):** explain the pipeline is not tuned to the 3 example
   clips — style-driven prompts work on any content (nature, urban, animals, people,
   sports, food, weather, technology).
7. **Wrap-up (15s):** modular repo (`pipeline/`, `models/`), public linux/amd64
   image (1.2GB) on the registry, plus a per-frame demo mode (`main.py`) as bonus tooling.

## Points to emphasise to the judges
- **100% harness-contract compliant:** `/input/tasks.json` → `/output/results.json`,
  exit 0, valid JSON, every style always present, linux/amd64, <10GB.
- **Fast & token-efficient:** one VLM call per video (not per frame), 4 parallel
  workers — the ~12-clip hidden set finishes far under 10 minutes.
- **Robust by design:** 3 fallback layers; a broken clip or dead URL never sinks
  the whole run.
- **Caption quality:** accurate to the specific content + tone match — exactly the
  two dimensions the LLM-Judge scores.
- **Generalisation:** nothing is hardcoded to the example clips; prompts are purely
  style-driven.

## Sample licensing notes
- 3 official example clips from the `amd-hackathon-clips` bucket (provided by organisers)
- Big Buck Bunny / Sintel © Blender Foundation — CC-BY 3.0 (used by the per-frame demo mode)
