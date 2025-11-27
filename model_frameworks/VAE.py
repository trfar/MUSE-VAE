import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

# Defining the ConvVAE Model
class ConvVAE(pl.LightningModule):
    def __init__(self, latent_dim=128,learning_rate=1e-4, n_mel_bins=128, frames=5187): 
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        # Encoder: Convolutional Layers
        self.encoder = nn.Sequential(
            # Encode Stack 1 (batch_size, 1, 128, 5187)
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), # Now 64x64
            nn.ReLU(),
            nn.InstanceNorm2d(32),
            # Encode Stack 2 (batch_size, 32, 64, 2594)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # Now 32x32
            nn.ReLU(),
            nn.InstanceNorm2d(64),
            # Encode Stack 3 (batch_size, 64, 32, 1297)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # Now 16x16 
            nn.ReLU(),
            nn.InstanceNorm2d(128),
            # Encode Stack 4 (batch_size, 128, 16, 648)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), ## Now 8x8       
            nn.ReLU(),
            nn.InstanceNorm2d(256),
            # Encode Stack 5 (batch_size, 256, 8, 324)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), # Now 4x4
            nn.ReLU(),
            nn.InstanceNorm2d(256),
            # Encode Stack 6 (batch_size, 256, 4, 162)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), # Now 2x2
            nn.ReLU(),
            nn.InstanceNorm2d(256),
            # Ending Stack (batch_size, 256, 2, 82)
        )
        
        # Fully Connected Layers for Mu and Logvar
        self.fc_mu = nn.Linear(256*2*82, self.latent_dim) 
        self.fc_logvar = nn.Linear(256*2*82, self.latent_dim) 

        # Linear projection
        self.decoder_fc = nn.Linear(self.latent_dim, 256 * 2 * 82)

        # Transposed-convolution decoder
        self.decoder = nn.Sequential(
            # Decode Stack 1 (batch_size, 256, 2, 82)
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.ReLU(),
            # Decode Stack 2 (batch_size, 256, 4, 164)
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.ReLU(),
            # Decode Stack 3 (batch_size, 256, 8, 328)
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            # Decode Stack 4 (batch_size, 128, 16, 656)
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            # Decode Stack 5 (batch_size, 64, 32, 1312)
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            # Decode Stack 6 (batch_size, 32, 64, 2624)
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            # Output Layer (batch_size, 1, 128, 5248)
            # Final Activation
            nn.Softplus()   # or nn.ReLU()
        )


    def encode(self, spectrograms):
        """Encodes the Input into Latent Space (Mean and Log Variance)."""
        audio_features = self.encoder(spectrograms) # (batch_size, 256, 2, 82)
        audio_features = audio_features.view(audio_features.size(0), -1)  # Flatten (batch_size, 256*2*82)
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
        audio_features = self.decoder_fc(audio_features)  # (batch_size, 256*2*82)
        audio_features = audio_features.view(batch_size, 256, 2, 82)  # Reshape to (batch_size, 256, 2, 82)
        # Generate Spectrogram Floats
        spectrograms = self.decoder(audio_features) # (batch_size, 1, 128, 5184)
        spectrograms = spectrograms[:, :, :, :5187]                    # crop to match input time length
        return spectrograms
    
    def forward(self, spectrograms):
        """Defines the Forward Pass."""
        batch_size = spectrograms.shape[0]
        mu, logvar = self.encode(spectrograms)
        audio_features = self.reparameterize(mu, logvar)
        spectrograms = self.decode(audio_features, batch_size)
        return spectrograms, mu, logvar
        
    def training_step(self, batch, batch_idx):
        """Training Logic for a Single Batch of Data."""
        ## Foward Pass through the Model
        spectrograms, mu, logvar = self(batch)
        ## Compute the Loss
        # MSE Loss
        mse_loss = F.mse_loss(torch.log1p(spectrograms), torch.log1p(batch))
        # KL Divergence Loss (Smoothing Latent Space)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).mean()  # KL Divergence Loss
        # KL Annealing
        anneal_epochs = 50
        kl_weight = min(self.current_epoch / anneal_epochs,1)  # Linear annealing
        kl_loss = kl_weight * kl_loss  # KL Annealing
        #Total Loss
        total_loss = mse_loss + kl_loss  # Total Loss

        ## Logging the Losses
        self.log('train_loss', total_loss)
        self.log('train_spectrogram_mse_log_loss', mse_loss)
        self.log('train_kl_loss', kl_loss)      

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation Logic for a Single Batch of Data."""
        ## Foward Pass through the Model
        spectrograms, mu, logvar = self(batch)
        ## Compute the Loss
        # MSE Loss
        mse_loss = F.mse_loss(torch.log1p(spectrograms), torch.log1p(batch))
        # KL Divergence Loss (Smoothing Latent Space)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).mean()  # KL Divergence Loss
        # KL Annealing
        anneal_epochs = 50
        kl_weight = min(self.current_epoch / anneal_epochs,1)  # Linear annealing
        kl_loss = kl_weight * kl_loss  # KL Annealing
        #Total Loss
        total_loss = mse_loss + kl_loss  # Total Loss

        ## Logging the Losses
        self.log('val_loss', total_loss)
        self.log('val_spectrogram_mse_loss', mse_loss)
        self.log('val_kl_loss', kl_loss)      

        return total_loss

    def configure_optimizers(self):
        """Optimizer + Learning Rate"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }
    
if __name__ == "__main__":
    cVAE=ConvVAE()
    
    