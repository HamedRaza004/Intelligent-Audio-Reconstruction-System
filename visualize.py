import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io

def make_waveform_plot(original_path: str, enhanced_path: str):
    orig, sr = librosa.load(original_path, sr=None)
    enh,  _  = librosa.load(enhanced_path, sr=None)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), facecolor='none')

    t_orig = np.linspace(0, len(orig)/sr, len(orig))
    t_enh  = np.linspace(0, len(enh)/sr,  len(enh))

    ax1.plot(t_orig, orig, color='#6B7FD4', linewidth=0.4, alpha=0.8)
    ax1.set_title("Original (noisy)", fontsize=10, color='gray')
    ax1.set_facecolor('none')
    ax1.tick_params(colors='gray')
    for spine in ax1.spines.values():
        spine.set_color('#333')

    ax2.plot(t_enh, enh, color='#4CAF82', linewidth=0.4, alpha=0.8)
    ax2.set_title("Enhanced (clean)", fontsize=10, color='gray')
    ax2.set_xlabel("Time (s)", color='gray')
    ax2.set_facecolor('none')
    ax2.tick_params(colors='gray')
    for spine in ax2.spines.values():
        spine.set_color('#333')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, transparent=True, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

def make_spectrogram_plot(original_path: str, enhanced_path: str):
    orig, sr = librosa.load(original_path, sr=None)
    enh,  _  = librosa.load(enhanced_path, sr=None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), facecolor='none')

    for ax, signal, title in [(ax1, orig, "Original"), (ax2, enh, "Enhanced")]:
        S = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
        librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
        ax.set_title(title, fontsize=10, color='gray')
        ax.set_facecolor('none')
        ax.tick_params(colors='gray')
        for spine in ax.spines.values():
            spine.set_color('#333')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, transparent=True, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf