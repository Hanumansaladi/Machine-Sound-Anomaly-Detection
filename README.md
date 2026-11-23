# 🎧 Machine Sound Anomaly Detection

### AI-Powered Industrial Machine Fault Detection Using Audio Analysis

<p align="center">
  <b>Detect abnormal machine sounds using Machine Learning and Audio Signal Processing.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Librosa-Audio%20Processing-purple" alt="Librosa">
  <img src="https://img.shields.io/badge/Status-Completed-success" alt="Status">
</p>

---

## 📌 Overview

Machine sounds change when a fault develops, and spotting those changes by ear is slow and unreliable.

This project analyzes a machine audio recording and classifies it as:

- 🟢 **Normal** – the machine is running with normal sound patterns.
- 🔴 **Abnormal** – the sound contains unusual patterns that may indicate a fault.

It combines audio signal processing with an **Isolation Forest** model, trained only on normal sounds, and provides a **Streamlit** dashboard where you can upload audio or record from a laptop microphone.

## 🏭 Dataset

Uses the **MIMII Dataset** (Malfunctioning Industrial Machine Investigation and Inspection).

| Setting | Value |
|---|---|
| Machine type | Fan |
| Machine ID | `id_00` |
| Sampling rate | 16 kHz |
| Duration | 10 seconds |
| Channels | 8 |
| SNR | 0 dB |

> The trained model currently supports **Fan `id_00` only**.

## 🎵 Audio Features

| Feature | Description |
|---|---|
| **RMS** | Average energy (loudness) of the signal |
| **Spectral Centroid** | Centre of mass of the spectrum (brightness) |
| **Spectral Bandwidth** | Spread of frequencies around the centroid |
| **Zero Crossing Rate** | How often the signal changes sign |

## 🤖 Model – Isolation Forest

An unsupervised anomaly detection algorithm that learns from normal machine sounds only, so no abnormal samples are needed for training.

1. Extract features from normal recordings.
2. Train the Isolation Forest model.
3. Score the normal validation recordings.
4. Set the decision threshold at the 95th percentile of those scores (saved value ≈ `0.5631495568`).
5. Compare new audio scores against the threshold to classify Normal or Abnormal.

```python
model = IsolationForest(contamination="auto", random_state=42)
model.fit(X_train)

threshold = np.percentile(normal_validation_scores, 95)
```

## 🔄 Workflow

```text
Audio Input (Upload / Microphone)
              |
      Feature Extraction
  RMS | Centroid | Bandwidth | ZCR
              |
       Isolation Forest
              |
   Anomaly Score vs Threshold
              |
      Normal  /  Abnormal
              |
   Streamlit UI + Report Download
```

## 🖥️ Dashboard

- Upload audio or record from the microphone
- Waveform and spectrogram visualization
- Normal / Abnormal prediction with anomaly score
- Downloadable prediction report

![Dashboard](screenshots/dashboard.png)

![Normal Prediction](screenshots/normal-result.png)

![Abnormal Prediction](screenshots/abnormal-result.png)

## 📊 Model Evaluation

| Metric | Result |
|---|---:|
| ROC-AUC | ≈ 0.826 |
| Accuracy | ≈ 0.62 |
| Normal Recall | ≈ 0.95 |
| Abnormal Recall | ≈ 0.50 |

These are exploratory results, not a final unbiased estimate. Validation on an independent test set is still needed.

## 🛠️ Technologies

Python · NumPy · Pandas · Librosa · Scikit-learn · Joblib · Streamlit · Plotly

## 📁 Project Structure

```text
Machine-Sound-Anomaly-Detection/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── 01_EDA.ipynb
├── app/
├── dataset/
├── models/
│   ├── isolation_forest.pkl
│   └── threshold.pkl
├── notebooks/
├── src/
├── reports/
└── results/
```

## ⚙️ Installation

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Machine-Sound-Anomaly-Detection

python -m venv dl-env
.\dl-env\Scripts\Activate.ps1    # Windows PowerShell

python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The dashboard opens in your browser.

## ⚠️ Limitations

- Supports only Fan `id_00`.
- Microphone recordings may differ from the original multichannel training data.
- Background noise and recording quality affect predictions.
- Detects that a sound is abnormal, not the exact fault.
- More independent testing is needed before industrial use.

## 🚀 Future Improvements

- Train models for more machine types and IDs.
- Improve abnormal sound detection.
- Explore deep learning-based anomaly detection.
- Improve microphone recording and preprocessing.
- Add more industrial audio datasets.
- Deploy the application online.

## 👨‍💻 Author

**Saladi Satya Hanuman** – B.Tech Student | AI & Machine Learning Enthusiast

- GitHub: [@hanumansaladi](https://github.com/hanumansaladi)
- LinkedIn: [Hanuman