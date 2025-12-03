import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


## GTZAN PyTorch Dataset
class GTZANDataset(Dataset):

    def __init__(self, dataframe):
        # Store dataframe
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        # Number of samples
        return len(self.df)

    def __getitem__(self, idx):
        # Load row
        row = self.df.iloc[idx]

        # Load mel spectrogram (.npy float array)
        mel = np.load(row["mel_spectrogram_path"]).astype(np.float32)

        # Expand to (1, 128, time)
        mel = np.expand_dims(mel, axis=0)

        # Genre string
        genre = row["genre"]

        return mel, genre


## Data Processing for GTZAN
class DataProcessing:

    def __init__(
        self,
        csv_path="/home/trfar/Documents/Advanced Machine Learning/MUSE-VAE/model_src/GTZAN_DATA_MASTER.csv",
        batch_size=64,
        test_size=0.15,
        val_size=0.15,
        num_workers=8,
        shuffle=True
    ):
        # Store params
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        # Load metadata CSV
        print(f"Loading metadata from: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)

        # Safety check
        required = {"genre", "mel_spectrogram_path"}
        if not required.issubset(self.df.columns):
            raise ValueError("CSV must contain 'genre' and 'mel_spectrogram_path' columns.")

        # Balanced split
        self.train_df, self.val_df, self.test_df = self.balanced_split()

        print(f"Train: {len(self.train_df)} | Val: {len(self.val_df)} | Test: {len(self.test_df)}")

        # Datasets
        self.train_dataset = GTZANDataset(self.train_df)
        self.val_dataset = GTZANDataset(self.val_df)
        self.test_dataset = GTZANDataset(self.test_df)

    ## Balanced train/val/test split by genre
    def balanced_split(self):

        genres = self.df["genre"].unique()

        train_list = []
        val_list = []
        test_list = []

        for g in genres:
            g_df = self.df[self.df["genre"] == g]

            train_g, temp_g = train_test_split(
                g_df,
                test_size=self.val_size + self.test_size,
                shuffle=True,
                random_state=42
            )

            relative_test_size = self.test_size / (self.test_size + self.val_size)

            val_g, test_g = train_test_split(
                temp_g,
                test_size=relative_test_size,
                shuffle=True,
                random_state=42
            )

            train_list.append(train_g)
            val_list.append(val_g)
            test_list.append(test_g)

        train_df = pd.concat(train_list).sample(frac=1).reset_index(drop=True)
        val_df   = pd.concat(val_list).sample(frac=1).reset_index(drop=True)
        test_df  = pd.concat(test_list).sample(frac=1).reset_index(drop=True)

        return train_df, val_df, test_df

    ## Return PyTorch DataLoaders
    def train_test_val_split(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return train_loader, val_loader, test_loader
