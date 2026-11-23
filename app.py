
import streamlit as st
import numpy as np
import soundfile as sf
import librosa
import joblib
import plotly.graph_objects as go
from pathlib import Path
from io import BytesIO
from datetime import datetime
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="MachineGuard AI",
    page_icon="🔊",
    layout="wide"
)

BASE = Path(__file__).parent

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = joblib.load(BASE / "models/isolation_forest.pkl")
    threshold = joblib.load(BASE / "models/threshold.pkl")
    return model, threshold

model, threshold = load_model()

# ---------------- PROFESSIONAL CSS ----------------

st.markdown("""
<style>
.stApp {
    background-color: #f7f7f2;
    color: #252525;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}

[data-testid="stSidebar"] {
    background-color: #efefe9;
    border-right: 1px solid #deded6;
}

.metric-card {
    background: #ffffff;
    padding: 20px;
    border: 1px solid #e5e5df;
    border-radius: 10px;
    margin-bottom: 12px;
}

.metric-label {
    color: #777777;
    font-size: 13px;
}

.metric-value {
    color: #222222;
    font-size: 25px;
    font-weight: 600;
}

h1, h2, h3 {
    color: #222222 !important;
}

.stButton button {
    background-color: #e8edf5;
    color: #263b58;
    border: 1px solid #d5dfeb;
    border-radius: 8px;
}

[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 1px solid #e5e5df;
    border-radius: 10px;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------

# ---------------- HEADER ----------------
st.title("🔊 Machine Sound Monitoring")
st.caption("AI-powered industrial machine anomaly detection")

st.divider()

# ---------------- TOP METRICS ----------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="metric-card">
    <div class="metric-label">SELECTED MACHINE</div>
    <div class="metric-value">Fan ID 00</div>
    </div>
    """, unsafe_allow_html=True)



# ---------------- INPUT SECTION ----------------
st.subheader("Audio Input")

tab1, tab2 = st.tabs(["📁 Upload Audio", "🎙️ Laptop Microphone"])

audio_data = None
sample_rate = None
input_type = None

with tab1:
    uploaded = st.file_uploader(
        "Upload MIMII WAV recording",
        type=["wav"]
    )

    if uploaded:
        audio_data, sample_rate = sf.read(
            BytesIO(uploaded.getvalue())
        )
        input_type = "MIMII Upload"

with tab2:
    st.warning(
        "Laptop microphone predictions are experimental. "
        "The model was trained on MIMII channel 6."
    )

    mic_audio = st.audio_input(
        "Record machine sound",
        sample_rate=16000
    )

    if mic_audio:
        audio_data, sample_rate = sf.read(
            BytesIO(mic_audio.getvalue())
        )
        input_type = "Laptop Microphone"

# ---------------- AUDIO ANALYSIS ----------------
if audio_data is not None:

    st.divider()
    st.subheader("Audio Analysis")

    if sample_rate != 16000:
        st.error("Audio must have a 16000 Hz sample rate.")
        st.stop()

    # Select channel
    if input_type == "MIMII Upload":
        if audio_data.ndim != 2 or audio_data.shape[1] < 7:
            st.error("Upload an 8-channel MIMII recording.")
            st.stop()

        signal = audio_data[:, 6]
        prediction_validity = True

    else:
        if audio_data.ndim == 2:
            signal = np.mean(audio_data, axis=1)
        else:
            signal = audio_data

        prediction_validity = False

    duration = len(signal) / sample_rate

    st.audio(
        uploaded if input_type == "MIMII Upload" else mic_audio
    )

    # Waveform
    st.markdown("#### Waveform")

    time_axis = np.arange(len(signal)) / sample_rate

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=signal,
        mode="lines",
        line=dict(color="#38bdf8", width=1)
    ))

    fig.update_layout(
        template="plotly_dark",
        height=280,
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        margin=dict(l=10, r=10, t=10, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Spectrogram
    st.markdown("#### Spectrogram")

    spectrum = np.abs(
        librosa.stft(signal.astype(np.float32), n_fft=1024, hop_length=512)
    ) ** 2

    mel = librosa.feature.melspectrogram(
        S=spectrum,
        sr=sample_rate,
        n_mels=64
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    spec_fig = go.Figure(data=go.Heatmap(
        z=mel_db,
        colorscale="Viridis",
        colorbar=dict(title="dB")
    ))

    spec_fig.update_layout(
        template="plotly_dark",
        height=320,
        xaxis_title="Time frames",
        yaxis_title="Mel frequency"
    )

    st.plotly_chart(spec_fig, use_container_width=True)

    # ---------------- FEATURES ----------------
    rms = np.mean(librosa.feature.rms(y=signal))
    centroid = np.mean(
        librosa.feature.spectral_centroid(y=signal, sr=sample_rate)
    )
    bandwidth = np.mean(
        librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate)
    )
    zcr = np.mean(np.abs(np.diff(np.sign(signal)))) / 2

    features = np.array([[
        rms, centroid, bandwidth, zcr
    ]])

    # ---------------- PREDICTION ----------------
    score = -model.score_samples(features)[0]
    prediction = "Abnormal" if score > threshold else "Normal"

    st.divider()
    st.subheader("AI Detection Results")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric("Anomaly Score", f"{score:.4f}")

    with r2:
        st.metric("Threshold", f"{threshold:.4f}")

    with r3:
        st.metric("Duration", f"{duration:.2f} sec")

    if prediction == "Abnormal":
        st.error("⚠️ ABNORMAL SOUND DETECTED")
    else:
        st.success("✅ NORMAL SOUND DETECTED")

    if not prediction_validity:
        st.warning(
            "Experimental result: this mono laptop recording "
            "has not been validated against the MIMII-trained model."
        )

    # ---------------- FEATURE CARDS ----------------
    st.subheader("Extracted Audio Features")

    f1, f2, f3, f4 = st.columns(4)

    f1.metric("RMS", f"{rms:.5f}")
    f2.metric("Spectral Centroid", f"{centroid:.2f}")
    f3.metric("Bandwidth", f"{bandwidth:.2f}")
    f4.metric("Zero Crossing Rate", f"{zcr:.5f}")

    # ---------------- DOWNLOAD REPORT ----------------
    report = pd.DataFrame([{
        "Machine": "Fan ID 00",
        "Input": input_type,
        "Prediction": prediction,
        "Anomaly Score": score,
        "Threshold": threshold,
        "RMS": rms,
        "Centroid": centroid,
        "Bandwidth": bandwidth,
        "ZCR": zcr,
        "Duration": duration
    }])

    st.download_button(
        "⬇️ Download Analysis Report",
        data=report.to_csv(index=False),
        file_name="machine_analysis_report.csv",
        mime="text/csv"
    )

st.divider()

st.caption(
    " Industrial Sound Anomaly Detection "
    
)
