## System Imports
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from argparse import ArgumentParser

## YAML Import
from ruamel.yaml import YAML

## PyTorch Imports
import torch

## Custom Imports
from DataLoader import DataProcessing
from model_frameworks.VAE import ConvVAE
from Vocoder import SimpleGriffinLimVocoder


def load_config(config_path):
    ## Load YAML Config File
    yaml = YAML()
    with open(config_path, "r") as f:
        config = yaml.load(f)
    return config


def build_conv_vae_from_checkpoint(checkpoint_path, config, device):
    ## Instantiate ConvVAE Model
    latent_dim = config.get("latent_dim", 128)
    learning_rate = config.get("learning_rate", 1e-4)

    model = ConvVAE(
        latent_dim=latent_dim,
        learning_rate=learning_rate
    ).to(device)

    ## Load Lightning Checkpoint And Map To ConvVAE
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    ## Strip "model." Prefix From Keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_key = k[len("model."):]
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    return model


def main():
    ## Argument Parser
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./model_frameworks/config.yaml",
        help="Path to config YAML"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional limit on number of test samples to process"
    )
    args = parser.parse_args()

    ## Resolve Project Root And GTZAN Paths
    this_file = os.path.abspath(__file__)
    model_src_dir = os.path.dirname(this_file)
    project_root = os.path.dirname(os.path.dirname(model_src_dir))

    gtzan_root = os.path.join(project_root, "GTZAN_Dataset")
    audio_out_dir = os.path.join(gtzan_root, "Generated_Audio")
    os.makedirs(audio_out_dir, exist_ok=True)

    ## Load Config
    config = load_config(args.config)

    ## Get Checkpoint Path From Config
    checkpoint_path = config.get("checkpoint_path", None)
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"checkpoint_path not set or does not exist in config: {checkpoint_path}"
        )

    ## Device Selection
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ## Build Model From Checkpoint
    print(f"Loading ConvVAE from checkpoint: {checkpoint_path}")
    model = build_conv_vae_from_checkpoint(checkpoint_path, config, device)

    ## Build Data Processing And Test Loader
    print("Building GTZAN test DataLoader")
    data_processing = DataProcessing()
    _, _, test_loader = data_processing.train_test_val_split()

    ## Instantiate Vocoder
    vocoder = SimpleGriffinLimVocoder(
        sample_rate=22050,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        device=device
    )

    ## Run Predictions On Test Set And Save Audio
    model.eval()
    sample_index = 0

    with torch.no_grad():
        for batch_idx, (x, genres) in enumerate(test_loader):
            x = x.to(device)
            recon_batch, mu, logvar = model(x)

            batch_size = recon_batch.size(0)

            for i in range(batch_size):
                if args.max_samples is not None and sample_index >= args.max_samples:
                    print(f"Reached max_samples limit: {args.max_samples}")
                    return

                ## Extract Single Reconstructed Mel
                mel_norm = recon_batch[i].detach().cpu()

                ## Ensure Shape (1, n_mels, T)
                if mel_norm.dim() == 4:
                    mel_norm = mel_norm.squeeze(0)
                if mel_norm.dim() == 2:
                    mel_norm = mel_norm.unsqueeze(0)

                ## Clamp To Valid Range
                mel_norm = torch.clamp(mel_norm, 0.0, 1.0)

                ## Get Genre Label
                genre_label = genres[i]
                if isinstance(genre_label, torch.Tensor):
                    genre_label = str(genre_label.item())

                ## Convert Mel To Audio
                audio = vocoder.mel_to_audio(mel_norm)

                ## Build Output Path
                file_name = f"sample_{sample_index:05d}_{genre_label}.wav"
                out_path = os.path.join(audio_out_dir, file_name)

                ## Save Audio
                vocoder.save_audio(audio, out_path)

                print(f"Saved {out_path}")
                sample_index += 1


if __name__ == "__main__":
    main()
