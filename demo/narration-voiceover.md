# Voiceover script for StyleCap-demo.mp4 (2:30, silent video)

Generate the voice with ElevenLabs, then mux it under the video (command at the
bottom). Timings below match the video segments exactly. Speak at a relaxed
pace; each block fits its window at roughly 150 words per minute. Paste one
block at a time into ElevenLabs so you can regenerate a single segment if the
timing drifts.

---

## 0:00 - 0:10 · Title

StyleCap: a video captioning agent built for the AMD Developer Hackathon, Act Two, running on Fireworks AI.

## 0:10 - 0:28 · The task

The judging harness hands the container a list of video URLs and four requested styles. For every clip, the agent must return one caption per style: formal, sarcastic, humorous tech, and humorous non-tech, as strict JSON, and exit cleanly. Here is what that looks like on the three official example clips.

## 0:28 - 0:52 · Clip 1, city traffic

First, an autumn boulevard with heavy traffic. The formal caption stays objective and factual. The sarcastic one goes dry: "Wow, cars driving on a road. Truly groundbreaking urban innovation." Then a programming joke about a traffic loop with zero memory leaks, and finally everyday humour about nobody knowing how to merge nicely. Same video, four completely different voices.

## 0:52 - 1:16 · Clip 2, kitten

Next, an orange kitten in a garden. Notice how every caption stays anchored to what is actually on screen: a kitten walking toward the camera through green foliage. The styles change the attitude, not the facts. That distinction is exactly what the judges score: accuracy to the content, and how well the tone lands.

## 1:16 - 1:40 · Clip 3, office

And an office worker typing at a desktop. The tech-humour caption reports keyboard input detected and CPU at one hundred percent focus, while the non-tech one notices the plant in the corner silently judging her. Nothing here is hardcoded to these examples; the prompts are purely style-driven, so the pipeline generalises to nature, sports, food, weather, whatever the hidden set brings.

## 1:40 - 1:58 · Architecture

Under the hood it is deliberately simple. Download the clip, sample eight evenly spaced frames with OpenCV, and make one multimodal call to Fireworks per video, not per frame. Clips are processed in parallel, and three layers of fallbacks guarantee that a dead URL or a broken clip never sinks the run, and no style ever comes back missing.

## 1:58 - 2:16 · Evaluation

Every design choice was measured, not guessed. An internal eval harness mirrors the judges' rubric and scored seven pipeline variants, including best-of-three sampling, a self-critique pass, and a Gemma 4 comparison. The shipped configuration won. We even learned that doubling the frame count actually hurts accuracy.

## 2:16 - 2:30 · Close

The image is public, one point two gigabytes, linux amd64, MIT licensed, and finishes the full hidden set well under the time budget. StyleCap: one video in, four voices out.

---

## Muxing the audio back in

Export the ElevenLabs audio as `narration.mp3`, then:

```
ffmpeg -i demo/StyleCap-demo.mp4 -i narration.mp3 -c:v copy -c:a aac -shortest demo/StyleCap-demo-final.mp4
```

If a segment runs long, regenerate just that block with a slightly faster
delivery, or pad the pauses between blocks in your audio editor; the video
gives each block a 2-3 second cushion.
