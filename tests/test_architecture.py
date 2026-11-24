"""
Architecture balance tests for ConvVAE.

Checks that the encoder/decoder round-trip preserves the spectrogram shape,
that the latent size follows the latent_dim knob, and that encoding is
per-sample (batch size does not change the result).

Run from the repo root:
    python tests/test_architecture.py
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model_frameworks.VAE import ConvVAE

INPUT_SHAPE = (1, 128, 216)  # (channels, mel bins, time)


def test_forward_preserves_input_shape():
    vae = ConvVAE(latent_dim=128)
    x = torch.rand(2, *INPUT_SHAPE)
    recon, mu, logvar = vae(x)
    assert recon.shape == x.shape, f"recon {tuple(recon.shape)} != input {tuple(x.shape)}"
    assert mu.shape == (2, 128)
    assert logvar.shape == (2, 128)


def test_decoder_output_is_finite_and_positive():
    vae = ConvVAE(latent_dim=128)
    recon, _, _ = vae(torch.rand(2, *INPUT_SHAPE))
    assert torch.isfinite(recon).all(), "reconstruction contains NaN/Inf"
    assert (recon >= 0).all(), "spectrogram output must be non-negative (Softplus)"


def test_latent_dim_knob_sizes_the_latent():
    for dim in (16, 1024):
        vae = ConvVAE(latent_dim=dim)
        _, mu, _ = vae(torch.rand(1, *INPUT_SHAPE))
        assert mu.shape == (1, dim), f"expected latent {dim}, got {mu.shape[1]}"


def test_encoding_is_independent_of_batch_size():
    vae = ConvVAE(latent_dim=128)
    x = torch.rand(1, *INPUT_SHAPE)
    mu1, _ = vae.encode(x)
    mu4, _ = vae.encode(x.repeat(4, 1, 1, 1))
    assert torch.allclose(mu1, mu4[0:1]), \
        "encoding of the same sample changed with batch size"


def test_decoder_projection_matches_latent_mapping():
    vae = ConvVAE(latent_dim=128)
    assert vae.decoder_fc.out_features == vae.enc_feat_dim, \
        "decoder_fc must project the latent back to the encoder feature size"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"[OK] {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {t.__name__}: {e}")
    if failed:
        sys.exit(1)
    print(f"All {len(tests)} tests passed.")
