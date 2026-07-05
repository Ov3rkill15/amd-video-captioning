# AMD Developer Hackathon: ACT II — Video Captioning

Track: **Video Captioning** (beginner-friendly, prompt: "have fun")
Hackathon: [lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii](https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii)
Timeline: 6–11 Juli 2026

## Ide
Pipeline yang mengekstrak frame (dan opsional audio) dari video, lalu menggunakan
VLM (Vision-Language Model) via Fireworks AI API untuk menghasilkan narasi per-frame.

## Stack
- Python 3.11
- OpenCV (ekstraksi frame), ffmpeg-python (ekstraksi audio)
- Fireworks AI API (VLM hosted, model default: `gemma-4-26b-a4b-it` — Gemma 4 26B multimodal)
- AMD Developer Cloud / ROCm / MI300X (opsional untuk eksperimen model lokal — credit belum aktif per 5 Juli)

## Kenapa Gemma
Track ini dipilih (selain track utama Video Captioning) karena ada bonus prize
"base use case for Gemma" dari Google DeepMind. Fireworks AI meng-host
`accounts/fireworks/models/gemma-4-26b-a4b-it` yang multimodal (vision + text,
262k context) — jadi satu pipeline ini otomatis qualify untuk track utama
DAN bonus prize Gemma, tanpa perlu setup terpisah.

## Status akses (update 2026-07-05)
- Akun AMD Developer Cloud: sudah dibuat, **credit GPU belum muncul**
- Fireworks AI API key: **sudah ada**, siap dipakai
- Karena captioning jalan lewat Fireworks API (bukan inference lokal), pipeline ini
  bisa dikembangkan & ditest penuh sekarang tanpa nunggu credit AMD cair.
  AMD Developer Cloud baru dibutuhkan nanti untuk eksperimen opsional di MI300X/ROCm.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # isi FIREWORKS_API_KEY
```

## Menjalankan pipeline

```bash
python main.py --input demo/samples/sample.mp4 --output demo/samples/output.json --fps 1
```

Opsional ekstrak audio juga:

```bash
python main.py --input demo/samples/sample.mp4 --extract-audio
```

## Menjalankan via Docker (submission wajib containerized)

```bash
docker build -t amd-video-captioning .
docker run --env-file .env -v "$PWD/demo:/app/demo" amd-video-captioning --input demo/samples/sample.mp4
```

## Struktur

```
src/
  pipeline/       # extract_frames.py, caption.py
  models/         # fireworks_client.py
  utils/          # io_utils.py
demo/
  samples/        # sample video, frame output, hasil JSON harian
  script.md       # naskah demo (isi H-1 sebelum deadline)
main.py           # CLI entrypoint
Dockerfile
```

## Progress log
<!-- update tiap hari sebagai bukti progres -->
- **2026-07-05**: Scaffolding awal — struktur pipeline, Fireworks client wrapper, Dockerfile, README.
- **2026-07-05**: Switch default VLM ke `gemma-4-26b-a4b-it` (qualify bonus prize Gemma). Fireworks API key sudah ada; AMD Developer Cloud credit masih pending.
