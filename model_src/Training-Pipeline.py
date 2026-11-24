# System Imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ruamel.yaml import YAML
from argparse import ArgumentParser

# PyTorch Imports
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

# Custom Imports
from model_src.DataLoader import DataProcessing
from model_frameworks.muse_vae import MUSE_VAE, SAVE_DIR

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
    data_processing = DataProcessing()
    train_loader, val_loader, test_loader = data_processing.train_test_val_split()

    ## Weights & Biases logger
    wandb_logger = WandbLogger(
        name=config.get('experiment_name', 'my_experiment'),
        project=config.get('wandb_project_name', 'MyProject'),
    )

    ## Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(SAVE_DIR, config.get('experiment_name', 'experiment'), 'checkpoints'),
        filename='epoch{epoch:03d}-val_loss{val_loss:.4f}',
        save_top_k=1,
        mode='min'
    )

    ## Early Stopping
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=config.get('patience', 25),
        verbose=True,
        mode='min'
    )

    ## Trainer
    trainer = Trainer(
        max_epochs=config.get('epochs', 200),
        accelerator='auto',
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

    ## Evaluating on Test Set
    trainer.test(model, dataloaders=test_loader, ckpt_path='best')

    ## Saving the Best Checkpoint to the YAML Config
    best_checkpoint_path = checkpoint_callback.best_model_path
    config['checkpoint_path'] = best_checkpoint_path
    print(f"Best CheckPoint Path: {best_checkpoint_path}")
    with open(args.config, 'w') as f:
        yaml.dump(config, f)
