""" This File Takes the GTZAN Datasent and Divdes Sound Clips into Mel Spectrograms and Saves a Master CSV File with Paths & Genre Labels"""
import os
import pandas as pd
import librosa
import numpy as np

class GTZAN_Processor():
    def __init__(self, 
                 root_dir="./",
                 file_output_dir="./",
                 output_csv_dir="GTZAN_DATA_MASTER.csv",
                 sample_rate=22050,
                 n_mels=128,
                 hop_length=512,
                 duration=30):
        self.root_dir = root_dir
        self.file_output_dir = file_output_dir
        self.output_csv_dir = output_csv_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.duration = duration  # in seconds
        self.audio_length = sample_rate * duration  # total samples

    def process_dataset(self):
        ## Get List of Genres and Start Metadata Collection
        genres = os.listdir(self.root_dir)
        metadata = []

        ## Loop Through Each Track in Each Genre and Compute Mel Spectrograms
        for genre in genres:
            genre_dir = os.path.join(self.root_dir, genre)
            if not os.path.isdir(genre_dir):
                continue
            for filename in os.listdir(genre_dir):
                # Confirm Wav Files
                if filename.endswith('.wav'):
                    print(f"Processing {filename} in genre {genre}...")
                    # Compute Mel Spectrograms
                    file_path = os.path.join(genre_dir, filename)
                    mel_spectrograms = self.compute_mel_spectrograms(file_path, n_second_cuts=5)
                    # Iterate through mel_spectrograms Dict and Save Each float array to File Output Directory
                    if mel_spectrograms is not None:
                        for spec in mel_spectrograms:
                            metadata.append({
                                'genre': genre,
                                'mel_spectrogram_path': spec['mel_spectrogram_path'],
                                'segment_index': spec['segment_index'],
                                'mel_spectrogram_shape': spec['mel_spectrogram_shape'],
                                'original_audio_path': file_path,

                            })
                    print(f"Finished processing {filename}.")
                    print()
        ## Convert Metadata to DataFrame and Save as CSV
        df = pd.DataFrame(metadata)
        df.to_csv(os.path.join(self.output_csv_dir), index=True)
        print(f"Metadata CSV saved to {self.output_csv_dir}")

    def compute_mel_spectrograms(self, file_path, n_second_cuts=5):
        try:
            ## Load Audio and Ensure the Correct Length
            y, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            if len(y) < self.audio_length:
                y = np.pad(y, (0, self.audio_length - len(y)), mode='constant')
            else:
                y = y[:self.audio_length]
            ## Divide Audio into Segments
            nCuts = self.duration // n_second_cuts
            mel_spectrograms = []
            for i in range(nCuts):
                ## Compute Mel Spectrogram for Each Segment
                start_sample = i * n_second_cuts * self.sample_rate
                end_sample = start_sample + n_second_cuts * self.sample_rate
                segment = y[start_sample:end_sample]
                mel_spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=self.n_mels, hop_length=self.hop_length)
                ## Convert to Decibels and Normalize Between 0 and 1
                mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
                # Standard Normilzation to Retain Amplitude Information
                mel_spectrogram_db_norm = np.clip((mel_spectrogram_db + 80) / 80, 0, 1)  # Normalize to [0, 1]
                os.makedirs(self.file_output_dir, exist_ok=True)
                save_path = os.path.join(self.file_output_dir, f"{os.path.basename(file_path).split('.')[0]}_segment_{i+1}.npy")
                # Save Mel Spectrogram as .npy File
                np.save(save_path, mel_spectrogram_db_norm)
                mel_spectrograms.append({'segment_index': i+1, 
                                         'mel_spectrogram_path': save_path,
                                         'mel_spectrogram_shape': mel_spectrogram_db.shape
                                         })
            return mel_spectrograms
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
        
if __name__ == "__main__":
    processor = GTZAN_Processor(
        root_dir="/home/trfar/Documents/Advanced Machine Learning/GTZAN_Dataset/genres_original/",
        file_output_dir="/home/trfar/Documents/Advanced Machine Learning/GTZAN_Dataset/Processed_Audio/",
        output_csv_dir="/home/trfar/Documents/Advanced Machine Learning/MUSE-VAE/model_src/GTZAN_DATA_MASTER.csv",
        sample_rate=22050,
        n_mels=128,
        hop_length=512,
        duration=30
    )
    processor.process_dataset()