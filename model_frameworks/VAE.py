import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl


## Convolutional Variational Autoencoder for 128x216 Mel Spectrograms
class ConvVAE(pl.LightningModule):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim

        ## Encoder (no hardcoded shapes, fully dynamic)
        self.encoder = nn.Sequential(
            # 128x216 -> 64x108
            nn.Conv2d(1, 32, kernel_size=3, stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.InstanceNorm2d(32),

            # 64x108 -> 32x54
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.InstanceNorm2d(64),

            # 32x54 -> 16x54
            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.InstanceNorm2d(128),

            # 16x27 -> 8x27
            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 1), padding=1),
            nn.ReLU(),
            nn.InstanceNorm2d(256),

            # 16x54 -> 16x54   (NO FURTHER DOWNSAMPLING)
            nn.Conv2d(256, 256, kernel_size=3, stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.InstanceNorm2d(256),
        )

        ## Decoder input shape (latent space projects here)
        self.enc_feat_h = 8
        self.enc_feat_w = 27
        self.enc_feat_channels = 256
        self.enc_feat_dim = self.enc_feat_channels * self.enc_feat_h * self.enc_feat_w

        ## Fully-connected layers for μ and log(σ²)
        self.fc_mu = nn.Linear(self.enc_feat_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.enc_feat_dim, self.latent_dim)
        ## Projection from latent space back into encoder shape
        self.decoder_fc = nn.Linear(self.latent_dim, self.enc_feat_dim)

        ## Decoder (projects the latent back to spectrogram shape)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0)),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=(2, 2), padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=(2, 2), padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=(2, 2), padding=1, output_padding=1),
            nn.Softplus()                     # positive output
        )

    def encode(self, spectrograms):
        """Encodes the Input into Latent Space (Mean and Log Variance)."""
        audio_features = self.encoder(spectrograms) # (batch, 256, 16, 54)
        audio_features = audio_features.view(audio_features.size(0), -1)  # Flatten (batch, 256*16*54)
        mu, logvar = self.fc_mu(audio_features), self.fc_logvar(audio_features) # (batch_size, latent_dim)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterize Trick to Sample from Latent Space."""
        std = torch.exp(0.5 * logvar)  # Log Variance to Std Dev
        eps = torch.randn_like(std)  # Sample from Normal Distribution
        return mu + eps * std  # Reparameterized Latent Variable

    def decode(self, audio_features, batch_size):
        """Decodes the Latent Variable to Audio (Spectrogram Floats)."""
        # First Fully Connected Layer
        audio_features = self.decoder_fc(audio_features)  # (batch, 256*8*27)
        audio_features = audio_features.view(batch_size, self.enc_feat_channels, self.enc_feat_h, self.enc_feat_w)  # Reshape to Dynamically Computed Shape
        # Generate Spectrogram Floats
        spectrograms = self.decoder(audio_features) # (batch, 1, 128, 216)
        spectrograms = spectrograms[:, :, :, :216]                    # crop to match input time length
        return spectrograms
    
    def forward(self, x):
        batch_size = x.size(0)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        spectrograms = self.decode(z, batch_size)
        return spectrograms, mu, logvar

    