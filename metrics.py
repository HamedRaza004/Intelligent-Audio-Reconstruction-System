import numpy as np
import librosa

def compute_snr(signal, noise):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2) + 1e-10
    return 10 * np.log10(signal_power / noise_power)

def compute_metrics(original_path: str, enhanced_path: str) -> dict:
    orig, sr1 = librosa.load(original_path, sr=None)
    enh, sr2  = librosa.load(enhanced_path, sr=None)

    min_len = min(len(orig), len(enh))
    orig, enh = orig[:min_len], enh[:min_len]

    noise = orig - enh
    snr_improvement = compute_snr(enh, noise) - compute_snr(orig, noise)

    rms_reduction = 20 * np.log10(
        (np.sqrt(np.mean(noise ** 2)) + 1e-10) /
        (np.sqrt(np.mean(orig ** 2)) + 1e-10)
    )

    return {
        "snr_improvement_db": round(abs(snr_improvement), 2),
        "noise_reduction_db": round(abs(rms_reduction), 2),
        "duration_sec": round(min_len / sr1, 2),
    }