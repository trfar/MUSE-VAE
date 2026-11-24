#!/usr/bin/env python
"""
Self-contained unit test for the Griffin-Lim vocoder.

Synthesizes a short tone chord, converts it to a normalized mel spectrogram
on the same scale the VAE is trained on (dB, clipped to [0, 1]), then
reconstructs a waveform and saves it to model_outputs/.

Run from the repo root:
    python tests/Vocoder_Unit_Test.py
"""
import os
import sys
import torch
import torchaudio
from argparse import ArgumentParser

# Vocoder.py lives in model_src/
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_src"))

from Vocoder import SimpleGriffinLimVocoder

SAMPLE_RATE = 22050
DURATION = 1.0  # seconds


def make_test_mel(sample_rate=SAMPLE_RATE, duration=DURATION,
                  n_mels=128, n_fft=2048, hop_length=512):
    """Build a [1, n_mels, T] mel spectrogram from a synthesized chord."""
    t = torch.linspace(0.0, duration, int(sample_rate * duration))
    freqs = torch.tensor([220.0, 277.18, 329.63])  # A3, C#4, E4
    wav = (torch.stack([torch.sin(2 * torch.pi * f * t) for f in freqs])
           .sum(dim=0) * 0.3)
    wav = wav.unsqueeze(0).unsqueeze(0)  # (1, 1, samples)

    mel_fn = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    mel_db = 10.0 * torch.log1p(mel_fn(wav))  # (1, n_mels, T)
    mel_db = mel_db - mel_db.max()            # ref = max, like GTZAN_Processor
    mel_norm = torch.clamp((mel_db + 80.0) / 80.0, 0.0, 1.0)
    return mel_norm


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--out_dir", type=str, default="model_outputs/vocoder_test",
        help="Directory for the reconstructed wav (default: model_outputs/vocoder_test)"
    )
    args = parser.parse_args()

    mel_norm = make_test_mel()
    print(f"Test mel shape: {tuple(mel_norm.shape)}")

    vocoder = SimpleGriffinLimVocoder(
        sample_rate=SAMPLE_RATE,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        device="cuda",  # falls back to CPU automatically
    )
    audio = vocoder.mel_to_audio(mel_norm)

    ## Sanity checks
    assert audio.dim() == 1, f"Expected 1-D waveform, got {audio.dim()}-D"
    assert torch.isfinite(audio).all(), "Reconstructed audio contains NaN/Inf"
    peak = audio.abs().max()
    assert peak > 0.0, "Reconstructed audio is silent"

    expected_samples = int(DURATION * SAMPLE_RATE)
    assert abs(len(audio) - expected_samples) < 2 * 512, (
        f"Unexpected audio length {len(audio)} vs ~{expected_samples}"
    )

    out_path = os.path.join(args.out_dir, "griffinlim_test.wav")
    vocoder.save_audio(audio, out_path)
    print(f"[OK] peak={peak:.3f} length={len(audio)} ({len(audio) / SAMPLE_RATE:.2f}s)")
    print(f"[OK] Saved to {out_path}")


if __name__ == "__main__":
    main()
