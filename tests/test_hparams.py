"""
Config-knob tests for MUSE_VAE (the shared Lightning wrapper).

Checks that values from the config dict actually reach the model:
learning rate, latent size, KL beta + annealing, and the warmup scheduler.

Run from the repo root:
    python tests/test_hparams.py
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model_frameworks.muse_vae import MUSE_VAE

X = torch.rand(1, 1, 128, 216)


def make_config(**overrides):
    cfg = {
        "experiment_name": "hparams_test",
        "lr": 3e-4,
        "latent_dim": 64,
        "anneal_epochs": 3,
        "KL_Beta": 0.5,
        "warmup": False,
        "warmup_steps": 0,
    }
    cfg.update(overrides)
    return cfg


def test_learning_rate_reaches_optimizer():
    model = MUSE_VAE(make_config())
    optimizer, _ = model.configure_optimizers()
    assert optimizer.param_groups[0]["lr"] == 3e-4


def test_latent_dim_reaches_model():
    model = MUSE_VAE(make_config(latent_dim=21))
    _, mu, _ = model(X)
    assert mu.shape == (1, 21)


def test_kl_is_zeroed_before_annealing():
    # Identical weights and identical reparameterization noise, so any loss
    # difference could only come from the KL term. At epoch 0 the annealing
    # scales it to zero, so both losses must be identical regardless of KL_Beta.
    torch.manual_seed(0)
    low = MUSE_VAE(make_config(KL_Beta=0.0))
    torch.manual_seed(0)
    high = MUSE_VAE(make_config(KL_Beta=10.0))
    low.current_epoch = high.current_epoch = 0
    torch.manual_seed(1)
    loss_low = low.training_step((X, "blues"), 0)
    torch.manual_seed(1)
    loss_high = high.training_step((X, "blues"), 0)
    assert torch.allclose(loss_low, loss_high), \
        "KL term should be annealed to zero at epoch 0"


def test_kl_beta_changes_loss_after_annealing():
    # Same setup, but past anneal_epochs: the KL divergence (always >= 0 and
    # > 0 for a random init) must now push the higher-beta loss up.
    torch.manual_seed(0)
    low = MUSE_VAE(make_config(KL_Beta=0.0))
    torch.manual_seed(0)
    high = MUSE_VAE(make_config(KL_Beta=10.0))
    low.current_epoch = high.current_epoch = 10
    torch.manual_seed(1)
    loss_low = low.training_step((X, "blues"), 0)
    torch.manual_seed(1)
    loss_high = high.training_step((X, "blues"), 0)
    assert loss_high > loss_low, \
        "KL_Beta should increase the loss once annealing has finished"


def test_warmup_scheduler_knob():
    _, off = MUSE_VAE(make_config(warmup=False)).configure_optimizers()
    assert len(off) == 1, "without warmup only the plateau scheduler should be returned"

    _, on = MUSE_VAE(make_config(warmup=True, warmup_steps=10)).configure_optimizers()
    assert len(on) == 2, "warmup should add a second, step-interval scheduler"
    lr_lambda = on[0]["scheduler"].lr_lambdas[0]
    assert lr_lambda(0) == 0.0
    assert lr_lambda(5) == 0.5
    assert lr_lambda(10) == 1.0


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
