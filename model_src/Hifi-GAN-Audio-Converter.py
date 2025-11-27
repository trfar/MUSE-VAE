import torch
from hifigan import Generator

generator = Generator().eval()
generator.load_state_dict(torch.load("hifigan_universal.pth"))
generator = generator.to("cuda")

def hifi_gan_audio_converter(mel_spectrogram):
    """
    Convert a mel spectrogram to audio using HiFi-GAN.

    Args:
        mel_spectrogram (torch.Tensor): A mel spectrogram tensor of shape (1, n_mels, time_steps).
    """
    with torch.no_grad():
        mel_spectrogram = mel_spectrogram.to("cuda")
        audio = generator(mel_spectrogram)
    return audio.cpu().squeeze()