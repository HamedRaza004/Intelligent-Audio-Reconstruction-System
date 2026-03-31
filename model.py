import torch
import numpy as np
import soundfile as sf
import resampy
from denoiser import pretrained
from denoiser.dsp import convert_audio

_model = None

def get_model():
    global _model
    if _model is None:
        _model = pretrained.dns64()
        _model.eval()
    return _model

def enhance_audio(input_path: str, output_path: str) -> dict:
    model = get_model()
    
    # Load audio using soundfile instead of torchaudio
    wav, sr = sf.read(input_path)
    
    # Convert to mono if stereo
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    
    # Resample to 16kHz if needed
    if sr != model.sample_rate:
        wav = resampy.resample(wav, sr, model.sample_rate)
    
    # Convert to tensor
    wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        enhanced = model(wav_tensor.unsqueeze(0))[0]
    
    # Save output
    enhanced_np = enhanced.squeeze().numpy()
    sf.write(output_path, enhanced_np, model.sample_rate)
    
    return {"sample_rate": model.sample_rate}