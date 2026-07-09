# Voiceover script for StyleCap-demo.mp4 (3:00, silent video)

Recorded 2026-07-10 as demo/vo/1..8.mp3 and muxed into
demo/StyleCap-demo-final.mp4. The video segments were re-timed to fit the
recorded blocks, so the timings below match the FINAL video. If you re-record
a block, keep it within its window and re-run the mux command at the bottom.

---

## 0:00 - 0:10 · Title

StyleCap: a video captioning agent built for the AMD Developer Hackathon, Act Two, running on Fireworks AI.

## 0:10 - 0:35 · The task

The judging harness hands the container a list of video URLs and four requested styles. For every clip, the agent must return one caption per style: formal, sarcastic, humorous tech, and humorous non-tech, as strict JSON, and exit cleanly. Here is what that looks like on the three official example clips.

## 0:35 - 1:03 · Clip 1, city traffic

First, an autumn boulevard with heavy traffic. The formal caption stays objective and factual. The sarcastic one goes dry: "Wow, cars driving on a road. Truly groundbreaking urban innovation." Then a programming joke about a traffic loop with zero memory leaks, and finally everyday humour about nobody knowing how to merge nicely. Same video, four completely different voices.

## 1:03 - 1:27 · Clip 2, kitten

Next, an orange kitten in a garden. Notice how every caption stays anchored to what is actually on screen: a kitten walking toward the camera through green foliage. The styles change the attitude, not the facts. That distinction is exactly what the judges score: accuracy to the content, and how well the tone lands.

## 1:27 - 1:55 · Clip 3, office

And an office worker typing at a desktop. The tech-humour caption reports keyboard input detected and CPU at one hundred percent focus, while the non-tech one notices the plant in the corner silently judging her. Nothing here is hardcoded to these examples; the prompts are purely style-driven, so the pipeline generalises to nature, sports, food, weather, whatever the hidden set brings.

## 1:55 - 2:20 · Architecture

Under the hood it is deliberately simple. Download the clip, sample eight evenly spaced frames with OpenCV, and make one multimodal call to Fireworks per video, not per frame. Clips are processed in parallel, and three layers of fallbacks guarantee that a dead URL or a broken clip never sinks the run, and no style ever comes back missing.

## 2:20 - 2:43 · Evaluation

Every design choice was measured, not guessed. An internal eval harness mirrors the judges' rubric and scored seven pipeline variants, including best-of-three sampling, a self-critique pass, and a Gemma 4 comparison. The shipped configuration won. We even learned that doubling the frame count actually hurts accuracy.

## 2:43 - 3:00 · Close

The image is public, one point two gigabytes, linux amd64, MIT licensed, and finishes the full hidden set well under the time budget. StyleCap: one video in, four voices out.

---

## Muxing the audio back in

With the 8 blocks saved as `demo/vo/1.mp3` .. `demo/vo/8.mp3`, each block is
delayed to its segment start (+0.4s lead) and mixed in one pass:

```
ffmpeg -y -i demo/StyleCap-demo.mp4 -i demo/vo/1.mp3 -i demo/vo/2.mp3 -i demo/vo/3.mp3 -i demo/vo/4.mp3 -i demo/vo/5.mp3 -i demo/vo/6.mp3 -i demo/vo/7.mp3 -i demo/vo/8.mp3 -filter_complex "[1:a]adelay=200|200[a1];[2:a]adelay=10400|10400[a2];[3:a]adelay=35400|35400[a3];[4:a]adelay=63900|63900[a4];[5:a]adelay=87900|87900[a5];[6:a]adelay=115900|115900[a6];[7:a]adelay=140900|140900[a7];[8:a]adelay=163900|163900[a8];[a1][a2][a3][a4][a5][a6][a7][a8]amix=inputs=8:normalize=0[aout]" -map 0:v -map "[aout]" -c:v copy -c:a aac -b:a 160k demo/StyleCap-demo-final.mp4
```
