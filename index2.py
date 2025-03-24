import streamlit as st
import torch
import torchaudio
import soundfile as sf
from demucs import pretrained
from demucs.apply import apply_model
import io
import base64
from pydub import AudioSegment
import os
import subprocess
import sys

# Function to dynamically install missing dependencies
def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        st.write(f"{package} not found. Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    finally:
        globals()[package] = __import__(package)

# Dynamically install required packages
install_and_import("ffmpeg_downloader")
import ffmpeg_downloader as ffdl

# Dynamically install ffmpeg and ffprobe if not available
def ensure_ffmpeg_installed():
    try:
        # Check if ffmpeg is already installed
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["ffprobe", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Install ffmpeg and ffprobe using ffmpeg_downloader
        st.write("FFmpeg/FFprobe not found. Installing FFmpeg and FFprobe...")
        ffdl.install()
        ffmpeg_path = ffdl.ffmpeg_path
        ffprobe_path = ffdl.ffprobe_path
        st.write(f"FFmpeg installed at: {ffmpeg_path}")
        st.write(f"FFprobe installed at: {ffprobe_path}")
        return ffmpeg_path, ffprobe_path
    else:
        # If ffmpeg and ffprobe are already installed, find their paths
        ffmpeg_path = subprocess.run(["which", "ffmpeg"], check=True, stdout=subprocess.PIPE).stdout.decode().strip()
        ffprobe_path = subprocess.run(["which", "ffprobe"], check=True, stdout=subprocess.PIPE).stdout.decode().strip()
        st.write(f"Using existing FFmpeg at: {ffmpeg_path}")
        st.write(f"Using existing FFprobe at: {ffprobe_path}")
        return ffmpeg_path, ffprobe_path

# Set up ffmpeg and ffprobe paths dynamically
def setup_ffmpeg_paths(ffmpeg_path, ffprobe_path):
    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffprobe = ffprobe_path

# Function to convert MP3 to WAV
def convert_mp3_to_wav(mp3_file):
    audio = AudioSegment.from_mp3(mp3_file)
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    buffer.seek(0)
    return buffer

# Function to convert audio buffer to downloadable link
def get_audio_download_link(audio_buffer, filename, text):
    b64 = base64.b64encode(audio_buffer.read()).decode()
    href = f'<a href="data:file/wav;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Ensure ffmpeg and ffprobe are installed and set up paths
ffmpeg_path, ffprobe_path = ensure_ffmpeg_installed()
setup_ffmpeg_paths(ffmpeg_path, ffprobe_path)

# Streamlit interface
st.title('Audio Source Separation')

# Upload file
uploaded_file = st.file_uploader("Choose an audio file...", type=['mp3', 'wav'])
if uploaded_file is not None:
    # Display file details
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
    st.write(file_details)

    try:
        # Load audio file
        if uploaded_file.type == 'audio/mpeg':  # Check if the file is an MP3
            st.write("Converting MP3 to WAV...")
            wav_buffer = convert_mp3_to_wav(uploaded_file)
            waveform, sr = torchaudio.load(wav_buffer)  # Load using torchaudio
        else:
            waveform, sr = torchaudio.load(uploaded_file)  # Load directly for WAV files

        waveform = waveform.unsqueeze(0)  # Add batch dimension

        # Load Demucs model
        st.write("Loading Demucs model...")
        model = pretrained.get_model('htdemucs')
        model.eval()
        model.cpu()

        # Source separation
        st.write("Running source separation...")
        with torch.no_grad():
            estimates = apply_model(model, waveform, shifts=1, split=True, overlap=0.25)

        # Process outputs
        vocals = estimates[0, 3]
        accompaniment = estimates[0, 0] + estimates[0, 1] + estimates[0, 2]
        accompaniment_np = accompaniment.cpu().numpy().T

        # Save accompaniment to buffer
        buffer = io.BytesIO()
        sf.write(buffer, accompaniment_np, sr, format='WAV')
        buffer.seek(0)

        # Create download link
        download_button = get_audio_download_link(buffer, 'accompaniment.wav', 'Download Accompaniment')
        st.markdown(download_button, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")