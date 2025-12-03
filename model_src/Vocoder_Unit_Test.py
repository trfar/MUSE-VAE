#!/usr/bin/env python

import os
import sys
import torch
import numpy as np
from argparse import ArgumentParser

# Ensure project imports work
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Vocoder import SimpleGriffinLimVocoder


def main():

    mel_path = "/home/trfar/Documents/Advanced Machine Learning/GTZAN_Dataset/Processed_Audio/blues_segment_1.npy"
    sample_rate = 22050

    if not os.path.exists(mel_path):
        raise FileNotFoundError(f"Mel file not found: {mel_path}")

    # Load mel (should be shape [128, T] or [1,128,T])
    mel = np.load(mel_path).astype(np.float32)

    if mel.ndim == 2:
        mel = np.expand_dims(mel, axis=0)  # → (1, 128, T)
    elif mel.ndim != 3 or mel.shape[0] != 1:
        raise ValueError(f"Expected shape (128,T) or (1,128,T). Got {mel.shape}")

    mel_tensor = torch.tensor(mel)

    # Instantiate Vocoder
    vocoder = SimpleGriffinLimVocoder(
        sample_rate=sample_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        device="cuda"
    )

    # Convert MEL → audio
    audio = vocoder.mel_to_audio(mel_tensor)

    # Build output directory to match your main script
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    generated_audio_dir = os.path.join(project_root, "GTZAN_Dataset", "Generated_Audio")
    os.makedirs(generated_audio_dir, exist_ok=True)

    # Output filename
    base = os.path.splitext(os.path.basename(mel_path))[0]
    out_path = os.path.join(generated_audio_dir, f"{base}_GriffinLim_Unit_Test.wav")

    # Save wav
    vocoder.save_audio(audio, out_path)

    print(f"[OK] Audio saved to: {out_path}")


if __name__ == "__main__":
    main()
