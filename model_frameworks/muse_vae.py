import os
import pandas as pd
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from model_frameworks.VAE import ConvVAE as TheModel

# Repo root (this file lives in <repo>/model_frameworks/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(REPO_ROOT, "TrainedModels")
os.makedirs(SAVE_DIR, exist_ok=True)


## Lightning wrapper around ConvVAE, shared by all pipelines
class MUSE_VAE(LightningModule):
    def __init__(self, config):
        super().__init__()
        # Save Config Hyperparameters for Checkpointing
        self.save_hyperparameters(config)
        # Primary config entries
        self.lr = config.get('lr', 1e-3)
        # Build model from model_class using config
        self.model = TheModel(
            latent_dim=config.get('latent_dim', 128),
        )
        # Storage for test results
        self.test_results = []
        # Warmup settings
        self.warmup = config.get("warmup", False)
        self.warmup_steps = config.get("warmup_steps", 0)
        self.anneal_epochs = config.get("anneal_epochs", 30)
        self.KL_beta = config.get("KL_Beta", .1)

    def forward(self, audio):
        """
        Forward pass: Returns a Tuple:
            (Spectrograms, mu, logvar).
        """
        return self.model(audio)

    def training_step(self, batch, batch_idx):
        ## Unpack batch
        x, _ = batch                       # x in [0,1], shape (B,1,128,216)

        ## Forward
        recon, mu, logvar = self(x)

        ## Clamp to valid range
        recon = torch.clamp(recon, 0.0, 1.0)
        x     = torch.clamp(x,     0.0, 1.0)

        ## Reconstruction loss DIRECTLY on normalized mel
        recon_loss = F.l1_loss(recon, x)

        ## KL as we set it before (beta-VAE)
        kl_elem = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_per_sample = kl_elem.sum(dim=1)
        kl_raw = kl_per_sample.mean()

        anneal_epochs = self.anneal_epochs
        max_beta = self.KL_beta
        beta = max_beta * min(self.current_epoch / anneal_epochs, 1.0)

        kl_loss = beta * kl_raw
        total_loss = recon_loss + kl_loss

        self.log("train_loss", total_loss)
        self.log("train_spectrogram_L1_loss", recon_loss)
        self.log("train_kl_loss", kl_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        recon, mu, logvar = self(x)

        recon = torch.clamp(recon, 0.0, 1.0)
        x     = torch.clamp(x,     0.0, 1.0)

        recon_loss = F.l1_loss(recon, x)

        kl_elem = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_per_sample = kl_elem.sum(dim=1)
        kl_raw = kl_per_sample.mean()

        anneal_epochs = self.anneal_epochs
        max_beta = self.KL_beta
        beta = max_beta * min(self.current_epoch / anneal_epochs, 1.0)

        kl_loss = beta * kl_raw
        total_loss = recon_loss + kl_loss

        self.log("val_loss", total_loss)
        self.log("val_spectrogram_L1_loss", recon_loss)
        self.log("val_kl_loss", kl_loss)

        return total_loss

    def test_step(self, batch, batch_idx):
        x, _ = batch
        spectrograms, mu, logvar = self(x)

        self.test_results.append({
            'spectrograms': spectrograms.detach().cpu(),}
            )

    def configure_optimizers(self):
        ## Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        ## Plateau scheduler (monitors val_loss)
        scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        schedulers = []

        # Optional warmup scheduler
        if self.hparams.get("warmup", False):

            def lr_lambda(step):
                warmup_steps = self.hparams.get("warmup_steps", 0)
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                return 1.0

            scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lr_lambda
            )

            # Step-wise warmup scheduler (no monitor needed)
            schedulers.append({
                "scheduler": scheduler_warmup,
                "interval": "step",
                "frequency": 1
            })

        # Epoch-wise ReduceLROnPlateau scheduler (monitors val_loss)
        schedulers.append({
            "scheduler": scheduler_plateau,
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1
        })

        # Lightning-friendly return: ([optimizers], [schedulers])
        return [optimizer], schedulers

    def on_test_epoch_end(self):
        """
        After Testing Set, Save Results Into Folder: <experiment_name>/eval
        """
        exp_name = self.hparams.get('experiment_name', 'experiment')
        results_dir = os.path.join(SAVE_DIR, exp_name, 'eval')
        os.makedirs(results_dir, exist_ok=True)

        out_path = os.path.join(results_dir, 'test_results.pkl')
        pd.to_pickle(self.test_results, out_path)
        print(f"Test results saved to {out_path}")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        spectrograms, mu, logvar = self(x)
        return spectrograms
