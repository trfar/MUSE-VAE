import os
import torch
import torchaudio

## Simple Griffin-Lim Vocoder
## Converts normalized mel spectrograms back to audio

class SimpleGriffinLimVocoder:
    def __init__(
        self,
        sample_rate=22050,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        device="cuda"
    ):
        ## Store parameters
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        ## Select device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        ## Mel inversion module
        self.inv_mel = torchaudio.transforms.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=0.0,
            f_max=sample_rate / 2.0
        ).to(self.device)

        ## Griffin-Lim phase reconstruction
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
            power=1.0,
            n_iter=64
        ).to(self.device)

    ## Denormalize mel spectrograms that were scaled to [0,1]
    def _denorm_log_mel(self, mel_norm):
        mel_db = mel_norm * 80.0 - 80.0
        mel_mag = 10.0 ** (mel_db / 20.0)
        return mel_mag

    ## Convert mel spectrogram to audio waveform
    def mel_to_audio(self, mel_norm):
        mel_norm = mel_norm.to(self.device)
        mel_mag = self._denorm_log_mel(mel_norm)
        linear_mag = self.inv_mel(mel_mag)
        linear_mag_2d = linear_mag.squeeze(0)
        audio = self.griffin_lim(linear_mag_2d)

        ## Center and normalize waveform so it is audible
        audio = audio - audio.mean()
        peak = audio.abs().max()
        if peak > 0:
            audio = audio / peak
            audio = audio * 0.95

        # print("Peak amplitude:", float(audio.abs().max()))
        return audio.detach().cpu()

    ## Save audio to disk
    def save_audio(self, audio, out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        torchaudio.save(out_path, audio, self.sample_rate)
