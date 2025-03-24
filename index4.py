import os
import streamlit as st
import torch
import torchaudio
import soundfile as sf
from demucs import pretrained
from demucs.apply import apply_model
import io
import base64
import tempfile
import numpy as np
from pydub import AudioSegment
from pydub.utils import which

# Try to use imageio-ffmpeg to get a static ffmpeg build
try:
    import imageio_ffmpeg
    # Get the path to the ffmpeg executable downloaded by imageio-ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    # Derive ffprobe's path based on ffmpeg's directory.
    # On many static builds, ffprobe is in the same folder as ffmpeg.
    ffprobe_path = os.path.join(os.path.dirname(ffmpeg_path), "ffprobe")
    if not os.path.exists(ffprobe_path):
        # On Windows, it might be named "ffprobe.exe"
        ffprobe_path = os.path.join(os.path.dirname(ffmpeg_path), "ffprobe.exe")
    # Set pydub to use these paths
    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffprobe = ffprobe_path
except Exception as e:
    st.warning(f"imageio-ffmpeg not available or failed: {e}. Falling back to system PATH lookup.")
    # Fallback: try to locate ffmpeg/ffprobe in PATH
    AudioSegment.converter = which("ffmpeg")
    AudioSegment.ffprobe = which("ffprobe")

# Debug prints (optional)
st.write("Using FFmpeg path:", AudioSegment.converter)
st.write("Using ffprobe path:", AudioSegment.ffprobe)

def get_audio_download_link(audio_buffer, filename, text):
    b64 = base64.b64encode(audio_buffer.read()).decode()
    href = f'<a href="data:file/wav;base64,{b64}" download="{filename}">{text}</a>'
    return href

st.title('Audio Source Separation')
uploaded_file = st.file_uploader("Choose an audio file...", type=['mp3', 'wav'])

if uploaded_file is not None:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
    st.write(file_details)

    suffix = ".mp3" if uploaded_file.type == "audio/mpeg" else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        waveform, sr = torchaudio.load(tmp_path)
    except Exception as e:
        st.warning("torchaudio.load failed, trying pydub as fallback...")
        try:
            audio = AudioSegment.from_file(tmp_path, format="mp3")
            samples = np.array(audio.get_array_of_samples())
            if audio.channels == 2:
                samples = samples.reshape((-1, 2)).T  # shape becomes (2, N)
            else:
                samples = samples.reshape(1, -1)
            waveform = torch.tensor(samples, dtype=torch.float32) / (2 ** 15)
            sr = audio.frame_rate
        except Exception as e2:
            st.error(f"Both torchaudio and pydub failed to load the file: {e2}")
            st.stop()

    waveform = waveform.unsqueeze(0)
    st.write("Loading Demucs model...")
    model = pretrained.get_model('htdemucs')
    model.eval()
    model.cpu()

    st.write("Running source separation...")
    with torch.no_grad():
        estimates = apply_model(model, waveform, shifts=1, split=True, overlap=0.25)

    accompaniment = estimates[0, 0] + estimates[0, 1] + estimates[0, 2]
    accompaniment_np = accompaniment.cpu().numpy().T

    buffer = io.BytesIO()
    sf.write(buffer, accompaniment_np, sr, format='WAV')
    buffer.seek(0)
    download_button = get_audio_download_link(buffer, 'accompaniment.wav', 'Download Accompaniment')
    st.markdown(download_button, unsafe_allow_html=True)
