#!/usr/bin/env python

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from argparse import ArgumentParser

from ruamel.yaml import YAML
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from lightning.pytorch import LightningModule
from model_src.DataLoader import DataProcessing
from model_frameworks.VAE import ConvVAE as ConvVAE


# Lightning wrapper, matching your training setup
class MUSE_VAE(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.lr = config.get("lr", 1e-3)
        self.model = ConvVAE(
            latent_dim=config.get("latent_dim", 128),
            learning_rate=config.get("learning_rate", 1e-4),
        )

    def forward(self, x):
        return self.model(x)  # returns (recon, mu, logvar)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./model_frameworks/config.yaml",
        help="Path to config YAML"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which dataset split to run PCA on"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on number of samples for PCA"
    )
    args = parser.parse_args()

    # Load config
    yaml = YAML()
    yaml.preserve_quotes = True
    with open(args.config, "r") as f:
        config = yaml.load(f)

    # Get checkpoint path
    checkpoint_path = config.get("checkpoint_path", None)
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"checkpoint_path missing or not found in config: {checkpoint_path}"
        )

    # Data processing / loaders
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

    train_loader, val_loader, test_loader = data_processing.train_test_val_split()

    if args.split == "train":
        loader = train_loader
    elif args.split == "val":
        loader = val_loader
    else:
        loader = test_loader

    # Load model from checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MUSE_VAE.load_from_checkpoint(
        checkpoint_path,
        config=config
    )
    model.to(device)
    model.eval()

    # Where to save results
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    AdvML_root = os.path.dirname(project_root)
    save_dir = os.path.join(project_root, "TrainedModels")
    save_dir_2 = os.path.join(AdvML_root, "TrainedModels")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_2, exist_ok=True)

    exp_name = config.get("experiment_name", "experiment")
    results_dir = os.path.join(save_dir, exp_name)
    results_dir_2 = os.path.join(save_dir_2, exp_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(results_dir_2, exist_ok=True)

    img_path = os.path.join(results_dir, f"latent_pca_{args.split}.png")
    img_path_2 = os.path.join(results_dir_2, f"latent_pca_{args.split}.png")
    csv_path_out = os.path.join(results_dir, f"latent_pca_{args.split}.csv")
    csv_path_out_2 = os.path.join(results_dir_2, f"latent_pca_{args.split}.csv")

    # Collect latent means and genres
    all_mu = []
    all_genres = []
    total_seen = 0
    max_samples = args.max_samples

    with torch.no_grad():
        for batch_idx, (x, genres) in enumerate(loader):
            x = x.to(device)

            recon, mu, logvar = model(x)  # mu: (B, latent_dim)
            mu_cpu = mu.detach().cpu().numpy()

            batch_size_current = mu_cpu.shape[0]

            for i in range(batch_size_current):
                if max_samples is not None and total_seen >= max_samples:
                    break

                all_mu.append(mu_cpu[i])
                # genres is a list of strings from GTZANDataset
                genre_label = genres[i]
                if not isinstance(genre_label, str):
                    try:
                        genre_label = genre_label.item()
                    except Exception:
                        genre_label = str(genre_label)
                all_genres.append(genre_label)
                total_seen += 1

            if max_samples is not None and total_seen >= max_samples:
                break

    if len(all_mu) == 0:
        raise RuntimeError("No samples collected for PCA.")

    latents = np.stack(all_mu, axis=0)  # (N, latent_dim)
    genres = np.array(all_genres)

    # PCA to 2D
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latents)

    var_ratio = pca.explained_variance_ratio_
    print(f"Explained variance ratio: PC1={var_ratio[0]:.4f}, PC2={var_ratio[1]:.4f}")

    # Save csv of points + genres (optional but useful)
    import pandas as pd
    df_out = pd.DataFrame({
        "pc1": latent_2d[:, 0],
        "pc2": latent_2d[:, 1],
        "genre": genres
    })
    df_out.to_csv(csv_path_out, index=False)
    df_out.to_csv(csv_path_out_2, index=False)
    print(f"[INFO] Saved latent PCA data to: {csv_path_out}")
    print(f"[INFO] Saved latent PCA data to: {csv_path_out_2}")

    # Plot scatter colored by genre
    unique_genres = sorted(list(set(genres)))
    cmap = plt.get_cmap("tab10", len(unique_genres))

    plt.figure(figsize=(8, 6))
    for idx, g in enumerate(unique_genres):
        mask = (genres == g)
        plt.scatter(
            latent_2d[mask, 0],
            latent_2d[mask, 1],
            s=40,
            alpha=0.7,
            color=cmap(idx),
            label=str(g)
        )

    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title(
        f"Latent PCA ({args.split} set)\n"
        f"PC1={var_ratio[0]:.2f}, PC2={var_ratio[1]:.2f}"
    )
    plt.legend(markerscale=2, fontsize=8, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(img_path, dpi=300)
    plt.savefig(img_path_2, dpi=300)
    plt.close()

    print(f"[INFO] Saved latent PCA plot to: {img_path}")
    print(f"[INFO] Saved latent PCA plot to: {img_path_2}")

if __name__ == "__main__":
    main()
