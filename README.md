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
- Fireworks AI API (VLM hosted, model default: `kimi-k2p6` — serverless, vision-capable)
- AMD Developer Cloud / ROCm / MI300X (opsional untuk eksperimen model lokal — credit belum aktif per 5 Juli)

## Kenapa bukan Gemma (untuk sekarang)
Track ini awalnya dipilih (selain track utama Video Captioning) karena ada bonus prize
"base use case for Gemma" dari Google DeepMind, lewat `accounts/fireworks/models/gemma-4-26b-a4b-it`
(multimodal, 262k context). Setelah ditest (2026-07-05), model ini di akun Fireworks kita
ternyata **hanya tersedia via Dedicated Deployment** (sewa GPU per jam, bukan serverless
pay-per-token) — API call ke endpoint chat completions biasa selalu 404.
Karena itu default VLM pipeline dipindah ke `kimi-k2p6` (serverless, vision-capable, sudah
terverifikasi jalan). Deployment Gemma jadi stretch goal opsional kalau ada waktu/budget GPU.

## Status akses (update 2026-07-05)
- Akun AMD Developer Cloud: sudah dibuat, **credit GPU belum muncul**
- Fireworks AI API key: **sudah ada**, siap dipakai
- Pipeline end-to-end **sudah ditest dan jalan** pakai video sintetis + model `kimi-k2p6`
  (lihat `demo/samples/output.json`)
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
docker run --rm --env-file .env -v "$(pwd)/demo:/app/demo" amd-video-captioning --input demo/samples/sample.mp4
```

> Kalau pakai Git Bash di Windows dan volume mount tidak muncul di host, jalankan dengan
> `MSYS_NO_PATHCONV=1` di depan `docker run` — Git Bash suka mengubah path container
> (`/app/demo`) jadi path Windows secara otomatis.

## Cek koneksi & akses model Fireworks

Sebelum jalanin pipeline penuh, verifikasi API key valid dan model VLM yang dikonfigurasi
di `.env` benar-benar accessible (serverless, bukan dedicated-deployment-only):

```bash
python scripts/check_model.py
```

Output `OK: '<model>' is accessible and supports image input.` berarti siap dipakai.
Kalau `FAIL`, script akan menampilkan daftar model serverless yang benar-benar accessible
di API key kamu.

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
- **2026-07-05**: Test end-to-end pakai video sintetis: `gemma-4-26b-a4b-it` ternyata 404 (cuma tersedia via Dedicated Deployment di akun kita, bukan serverless). Switch default VLM ke `kimi-k2p6` (serverless, vision-capable). Juga fix bug: response kimi-k2 berisi reasoning trace mentah tanpa `reasoning_effort: "none"` — sudah ditambahkan di `fireworks_client.py`. Pipeline sekarang jalan bersih end-to-end.
- **2026-07-05**: Docker image di-build & ditest (`docker build` + `docker run` dengan volume mount) — hasil caption identik dengan run lokal. Tambah `scripts/check_model.py` untuk verifikasi cepat API key + model access tanpa proses video penuh.
