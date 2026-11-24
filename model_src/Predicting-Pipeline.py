#!/usr/bin/env python

## System Imports
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from argparse import ArgumentParser

from ruamel.yaml import YAML
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

## Project Imports
from model_src.DataLoader import DataProcessing
from model_frameworks.VAE import ConvVAE as ConvVAE  # same class you used in training
from lightning.pytorch import LightningModule


## Lightning Wrapper (same structure as your MUSE_VAE in training)
class MUSE_VAE(LightningModule):
    def __init__(self, config):
        super().__init__()
        # Save config for checkpointing
        self.save_hyperparameters(config)

        # Learning rate
        self.lr = config.get("lr", 1e-3)

        # Build ConvVAE model
        self.model = ConvVAE(
            latent_dim=config.get("latent_dim", 128),
            learning_rate=config.get("learning_rate", 1e-4),
        )

    def forward(self, x):
        # Forward through ConvVAE
        return self.model(x)


## Helper function to plot one pair to the PDF
def plot_pair_to_pdf(original, reconstructed, genre, index, pdf):
    # original, reconstructed: 2D numpy arrays [128, 216]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Original
    ax0 = axes[0]
    im0 = ax0.imshow(original, aspect="auto", origin="lower")
    ax0.set_title(f"Original (Genre: {genre})")
    ax0.set_xlabel("Time")
    ax0.set_ylabel("Mel Bins")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    # Reconstructed
    ax1 = axes[1]
    im1 = ax1.imshow(reconstructed, aspect="auto", origin="lower")
    ax1.set_title("Reconstructed")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Mel Bins")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    fig.suptitle(f"Sample {index}", fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


if __name__ == "__main__":
    ## Argument Parser
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./model_frameworks/config.yaml",
        help="Path to config YAML"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=64,
        help="Maximum number of test samples to visualize"
    )
    args = parser.parse_args()

    ## YAML Loader
    yaml = YAML()
    yaml.preserve_quotes = True

    ## Load Config
    with open(args.config, "r") as f:
        config = yaml.load(f)

    ## Get checkpoint path from config
    checkpoint_path = config.get("checkpoint_path", None)
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"checkpoint_path missing or not found in config: {checkpoint_path}"
        )

    ## Data Processing (GTZAN)
    # Expecting CSV path in config or defaulting
    csv_path = config.get(
        "csv_path",
        "./model_src/GTZAN_DATA_MASTER.csv"
    )

    batch_size = config.get("batch_size", 64)

    data_processing = DataProcessing(
        csv_path=csv_path,
        batch_size=batch_size,
        test_size=config.get("test_size", 0.15),
        val_size=config.get("val_size", 0.15),
        num_workers=config.get("num_workers", 8),
        shuffle=False
    )

    # Train, val, test loaders (we only need test)
    _, _, test_loader = data_processing.train_test_val_split()

    ## Load Model from Checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MUSE_VAE.load_from_checkpoint(
        checkpoint_path,
        config=config
    )
    model.to(device)
    model.eval()

    ## Set up PDF output path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    AdvML_root = os.path.dirname(project_root)
    save_dir_project = os.path.join(project_root, "TrainedModels")
    os.makedirs(save_dir_project, exist_ok=True)

    exp_name = config.get("experiment_name", "experiment")
    results_dir = os.path.join(save_dir_project, exp_name)
    os.makedirs(results_dir, exist_ok=True)

    pdf_path_1 = os.path.join(results_dir, "Mel_Reconstruction.pdf")
    pdf_path_2 = os.path.join(AdvML_root, "TrainedModels", f"{exp_name}_Mel_Reconstruction.pdf")

    ## Run through test set and save comparisons
    max_examples = args.max_examples
    example_count = 0

    with PdfPages(pdf_path_1) as pdf:
        with torch.no_grad():
            for batch_idx, (x, genres) in enumerate(test_loader):
                # x: (B, 1, 128, 216)
                x = x.to(device)

                # Forward pass through MUSE_VAE -> ConvVAE
                recon, mu, logvar = model(x)

                # Move to CPU for plotting
                x_cpu = x.detach().cpu()
                recon_cpu = recon.detach().cpu()

                batch_size_current = x_cpu.size(0)

                for i in range(batch_size_current):
                    if example_count >= max_examples:
                        break

                    # Extract spectrograms [128, 216]
                    original = x_cpu[i, 0].numpy()
                    reconstructed = recon_cpu[i, 0].numpy()
                    genre = genres[i]

                    # Convert genre from tensor if needed
                    if not isinstance(genre, str):
                        try:
                            genre = genre.item()
                        except Exception:
                            genre = str(genre)

                    # Plot to PDF
                    plot_pair_to_pdf(
                        original=original,
                        reconstructed=reconstructed,
                        genre=genre,
                        index=example_count,
                        pdf=pdf
                    )

                    # Also save to secondary location
                    plot_pair_to_pdf(
                        original=original,
                        reconstructed=reconstructed,
                        genre=genre,
                        index=example_count,
                        pdf=PdfPages(pdf_path_2)
                    )

                    example_count += 1

                if example_count >= max_examples:
                    break

    print(f"[INFO] Reconstruction PDF saved to: {pdf_path_1}")
    print(f"[INFO] Reconstruction PDF also saved to: {pdf_path_2}")