import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def plot_wav(filepath):
    sr, audio = wavfile.read("test.wav")

    # Convert stereo to mono if needed
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    time = np.arange(len(audio)) / sr

    plt.figure(figsize=(14, 4))
    plt.plot(time, audio, linewidth=0.6)
    plt.title("Raw Waveform")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

    return audio, sr

def plot_sin_abs(audio, sr):
    transformed = np.abs(np.sin(audio))

    time = np.arange(len(transformed)) / sr

    plt.figure(figsize=(14, 4))
    plt.plot(time, transformed, linewidth=0.6)
    plt.title("|sin(audio)| Waveform")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

    return transformed

def plot_rms(audio, sr, window_ms=25):
    window_size = int(sr * window_ms / 1000)

    # RMS calculation
    rms = np.sqrt(
        np.convolve(audio ** 2,
                    np.ones(window_size) / window_size,
                    mode="valid")
    )

    time = np.arange(len(rms)) / sr

    plt.figure(figsize=(14, 4))
    plt.plot(time, rms, linewidth=1.2)
    plt.title(f"RMS Smoothed Signal ({window_ms} ms window)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Energy")
    plt.tight_layout()
    plt.show()

    return rms

audio, sr = plot_wav("test.wav")

sin_audio = plot_sin_abs(audio, sr)

rms_audio = plot_rms(audio, sr, window_ms=30)
