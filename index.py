import streamlit as st
import torch
import torchaudio
import soundfile as sf
from demucs import pretrained
from demucs.apply import apply_model
import io
import base64

# Function to convert audio buffer to downloadable link
def get_audio_download_link(audio_buffer, filename, text):
    b64 = base64.b64encode(audio_buffer.read()).decode()
    href = f'<a href="data:file/wav;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Streamlit interface
st.title('Audio Source Separation')

# Upload file
uploaded_file = st.file_uploader("Choose an audio file...", type=['mp3', 'wav'])

if uploaded_file is not None:
    # Display file details
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
    st.write(file_details)

    # Load audio file
    waveform, sr = torchaudio.load(uploaded_file)  # Load using torchaudio
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
