# System Imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ruamel.yaml import YAML
import pandas as pd
from argparse import ArgumentParser

# PyTorch Imports
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

# Custom Imports
from model_frameworks.VAE import ConvVAE as TheModel

# Setting Propper Path for Saving Models
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
save_dir = os.path.join(project_root, "TrainedModels")
os.makedirs(save_dir, exist_ok=True)

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
            learning_rate=config.get('learning_rate', 1e-4),
            n_mel_bins=config.get('n_mel_bins', 128),
            frames=config.get('frames', 5187)
        )
        # Storage for test results
        self.test_results = []
        # Warmup settings
        self.warmup = config.get("warmup", False)
        self.warmup_steps = config.get("warmup_steps", 0)

    def forward(self, audio):
        """
        Forward pass: Returns a Tuple:
            (Spectrograms, mu, logvar).
        """
        return self.model(audio)

    def training_step(self, batch, batch_idx):
        """Training Logic for a Single Batch of Data."""
        spectrograms, mu, logvar = self(batch)
        eps = 1e-5
        # Clamp negatives
        spectrograms = torch.clamp(spectrograms, min=0)
        batch = torch.clamp(batch, min=0)
        # Reconstruction loss (L1 on log scale)
        recon_loss = F.l1_loss(torch.log1p(spectrograms + eps),
                            torch.log1p(batch + eps))
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).mean()
        # KL annealing
        anneal_epochs = 50
        kl_weight = min(self.current_epoch / anneal_epochs, 1)
        kl_loss = kl_weight * kl_loss
        # Total VAE loss
        total_loss = recon_loss + kl_loss

        ## Logging the Losses
        self.log('train_loss', total_loss)
        self.log('train_spectrogram_L1_loss', recon_loss)
        self.log('train_kl_loss', kl_loss)      

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation Logic for a Single Batch of Data."""
        ## Foward Pass through the Model
        spectrograms, mu, logvar = self(batch)
        eps = 1e-5
        # Clamp negatives
        spectrograms = torch.clamp(spectrograms, min=0)
        batch = torch.clamp(batch, min=0)
        ## Compute the Loss
        # L1 Loss
        recon_loss = F.l1_loss(torch.log1p(spectrograms + eps),
                       torch.log1p(batch + eps))
        # KL Divergence Loss (Smoothing Latent Space)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).mean()  # KL Divergence Loss
        # KL Annealing
        anneal_epochs = 50
        kl_weight = min(self.current_epoch / anneal_epochs,1)  # Linear annealing
        kl_loss = kl_weight * kl_loss  # KL Annealing
        #Total Loss
        total_loss = recon_loss + kl_loss  # Total Loss

        ## Logging the Losses
        self.log('val_loss', total_loss)
        self.log('val_spectrogram_L1_loss', recon_loss)
        self.log('val_kl_loss', kl_loss)      

        return total_loss

    def test_step(self, batch, batch_idx):
        spectrograms, mu, logvar= self(batch)
        self.test_results.append({
            'spectrograms': spectrograms.detach().cpu(),}
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        ## Reduce-On-Plateau Scheduler
        scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        ## Warmup Scheduler
        if self.hparams.get("warmup", False):

            def lr_lambda(step):
                warmup_steps = self.hparams.get("warmup_steps", 0)
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                return 1.0

            scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_lambda
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": [
                    {
                        "scheduler": scheduler_warmup,
                        "interval": "step",   
                        "frequency": 1,
                    },
                    {
                        "scheduler": scheduler_plateau,
                        "monitor": "val_loss",
                        "interval": "epoch",              
                        "frequency": 1,
                    }
                ]
            }

        ## No Warmup Return
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler_plateau,
                "monitor": "val_loss",
                "interval": "epoch"
            }
        }

    def on_test_epoch_end(self):
        """
        After Testing Set, Save Results Into Folder: <experiment_name>/eval
        """
        exp_name = self.hparams.get('experiment_name', 'experiment')
        results_dir = os.path.join(save_dir, exp_name, 'eval')
        os.makedirs(results_dir, exist_ok=True)

        out_path = os.path.join(results_dir, 'test_results.pkl')
        pd.to_pickle(self.test_results, out_path)
        print(f"Test results saved to {out_path}")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        spectrograms, mu, logvar = self(batch)
        return spectrograms

if __name__ == '__main__':
    ## Arguments for Bash Scripts
    # Parsing Command Lines
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='./model_frameworks/config.yaml', help='Path to config YAML') 
    # Setting up YAML to keep Headers and Comments
    yaml = YAML()
    yaml.preserve_quotes = True
    args = parser.parse_args()
    
    ## Loading Config
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    ## Prepare data
    data_processing = DataProcessing(root_dir=config.get('root_dir', './'),
                                     data_dir=config.get('data_dir', 'data'),
                                     args=config)
    train_loader, val_loader, test_loader = data_processing.train_test_val_split(
        batch_size=config.get('batch_size', 64)
    )

    ## Weights & Biases logger
    wandb_logger = WandbLogger(
        name=config.get('experiment_name', 'my_experiment'),
        project=config.get('wandb_project_name', 'MyProject'),
    )

    ## Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(save_dir, config.get('experiment_name', 'experiment'), 'checkpoints'),
        filename='epoch{epoch:03d}-val_loss{val_loss:.4f}',
        save_top_k=1,
        mode='min'
    )

    ## Early Stopping
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=config.get('patience', 15),
        verbose=True,
        mode='min'
    )

    ## Trainer
    trainer = Trainer(
        max_epochs=config.get('epochs', 150),
        accelerator='gpu',
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        check_val_every_n_epoch=config.get('check_val_every_n_epoch', 1),
        log_every_n_steps=5,
    )

    ## Instantiate the LightningModule
    model = MUSE_VAE(config)

    ## Fit
    trainer.fit(model, train_loader, val_loader)

    ## Evaluating Test Set Every N Epochs
    test_freq = config.get('test_every_n_epochs', 50)
    total_epochs = config.get('epochs', 150)
    # Testing the Model
    for epoch_mark in range(test_freq, total_epochs + 1, test_freq):
        print(f"\nRunning test at epoch {epoch_mark}")
        trainer.test(model, dataloaders=test_loader, ckpt_path='best')

    ## Saving the Best Checkpoint to the YAML Config
    best_checkpoint_path = checkpoint_callback.best_model_path
    config['checkpoint_path'] = best_checkpoint_path
    print(f"Best CheckPoint Path: {best_checkpoint_path}")
    with open(args.config, 'w') as f:
        yaml.dump(config, f)