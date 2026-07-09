"""StyleCap — live Streamlit demo for the AMD ACT II Track 2 captioning agent.

This wraps the SAME pipeline the submission agent (agent.py) runs, so the hosted
demo faithfully represents the judged behavior:

    resolve_video -> sample_frames_b64 -> generate_styled_captions

Deployed on Streamlit Community Cloud straight from the public GitHub repo.
Locally:  FIREWORKS_API_KEY=... streamlit run streamlit_app.py
"""

import base64
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

# --- credentials -----------------------------------------------------------
# Streamlit Cloud has no .env; it injects st.secrets. Copy either the plain key
# or its base64 form into the environment so the unchanged resolve_api_key() in
# src/models/fireworks_client.py picks it up. load_dotenv() covers local runs.
load_dotenv(REPO_ROOT / ".env")
for _name in ("FIREWORKS_API_KEY", "FIREWORKS_API_KEY_B64", "FIREWORKS_VLM_MODEL"):
    try:
        if _name in st.secrets and not os.environ.get(_name):
            os.environ[_name] = str(st.secrets[_name])
    except Exception:
        # st.secrets raises if no secrets file exists at all (local dev) — ignore.
        pass

from agent import resolve_video  # noqa: E402
from src.models.fireworks_client import FireworksVLMClient  # noqa: E402
from src.pipeline.extract_frames import sample_frames_b64  # noqa: E402
from src.pipeline.style_caption import generate_styled_captions  # noqa: E402

ALL_STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
STYLE_LABELS = {
    "formal": "Formal",
    "sarcastic": "Sarcastic",
    "humorous_tech": "Humorous (tech)",
    "humorous_non_tech": "Humorous (non-tech)",
}
SAMPLE_CLIPS = {
    "Big Buck Bunny (720p, CC-BY)": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_1MB.mp4",
    "Sintel (360p, CC-BY)": "https://test-videos.co.uk/vids/sintel/mp4/h264/360/Sintel_360_10s_1MB.mp4",
}


@st.cache_resource
def get_client() -> FireworksVLMClient:
    return FireworksVLMClient()


def run_pipeline(video_ref: str, styles: list[str]) -> tuple[list[str], dict[str, str]]:
    """resolve -> sample 8 frames -> one multimodal call -> styled captions."""
    client = get_client()
    with tempfile.TemporaryDirectory() as work_dir:
        video_path = resolve_video(video_ref, work_dir, "demo")
        frames = sample_frames_b64(video_path, num_frames=8)
        captions = generate_styled_captions(client, frames, styles)
    return frames, captions


st.set_page_config(page_title="StyleCap — one video, four voices", page_icon="🎬", layout="centered")

st.title("🎬 StyleCap")
st.caption("One video clip, four caption voices. AMD ACT II — Track 2 (Video Captioning).")

try:
    _model = get_client().model.split("/")[-1]
    st.markdown(f"**Vision model:** `{_model}` (Fireworks AI) &nbsp;•&nbsp; 8 sampled frames per clip")
except Exception:
    st.warning(
        "No Fireworks API key configured. On Streamlit Cloud, set `FIREWORKS_API_KEY_B64` "
        "in the app's Secrets. Locally, run with `FIREWORKS_API_KEY=... streamlit run streamlit_app.py`."
    )

st.divider()

source = st.radio(
    "Video source",
    ["Sample clip", "Paste a video URL", "Upload a file"],
    horizontal=True,
)

video_ref: str | None = None
uploaded_tmp: str | None = None

if source == "Sample clip":
    choice = st.selectbox("Pick a Creative Commons sample", list(SAMPLE_CLIPS))
    video_ref = SAMPLE_CLIPS[choice]
    st.video(video_ref)
elif source == "Paste a video URL":
    url = st.text_input("Direct link to an .mp4 (http/https)")
    if url.strip():
        video_ref = url.strip()
        st.video(video_ref)
else:
    up = st.file_uploader("Upload a short clip", type=["mp4", "webm", "mov", "mkv", "avi"])
    if up is not None:
        suffix = Path(up.name).suffix or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(up.read())
        tmp.close()
        uploaded_tmp = tmp.name
        video_ref = tmp.name
        st.video(up)

styles = st.multiselect(
    "Caption styles",
    ALL_STYLES,
    default=ALL_STYLES,
    format_func=lambda s: STYLE_LABELS.get(s, s),
)

if st.button("Generate captions", type="primary", disabled=not (video_ref and styles)):
    try:
        with st.spinner("Sampling frames and writing captions..."):
            frames, captions = run_pipeline(video_ref, styles)
    except Exception as e:  # noqa: BLE001 — surface any pipeline/API error to the UI
        st.error(f"{type(e).__name__}: {e}")
    else:
        st.subheader("Sampled frames")
        cols = st.columns(min(4, len(frames)))
        for i, f in enumerate(frames):
            cols[i % len(cols)].image(base64.b64decode(f), use_container_width=True)

        st.subheader("Captions")
        for style in styles:
            st.markdown(f"**{STYLE_LABELS.get(style, style)}**")
            st.write(captions.get(style, ""))
    finally:
        if uploaded_tmp:
            Path(uploaded_tmp).unlink(missing_ok=True)

st.divider()
st.caption(
    "Runs the same pipeline as the containerized submission agent "
    "(`agent.py`): OpenCV frame sampling + one multimodal Fireworks call per clip."
)
