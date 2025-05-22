# spectrogram_gen.py
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

def audio_to_rgb_spectrogram(wav_path, output_dir, n_fft=1024, hop_length=256):
    y, sr = librosa.load(wav_path, sr=44100)
    duration = 5  # seconds per chunk
    samples_per_clip = duration * sr
    total_clips = len(y) // samples_per_clip

    os.makedirs(output_dir, exist_ok=True)

    for i in range(total_clips):
        start = i * samples_per_clip
        end = start + samples_per_clip
        clip = y[start:end]

        S = librosa.stft(clip, n_fft=n_fft, hop_length=hop_length)
        S_mag = np.abs(S)
        S_db = librosa.amplitude_to_db(S_mag, ref=np.max)
        S_db_norm = (S_db + 80) / 80  # normalize to [0,1]

        # Plot and save as RGB
        plt.figure(figsize=(10, 5))
        plt.axis('off')
        plt.imshow(S_db_norm, aspect='auto', origin='lower', cmap='viridis')
        plt.tight_layout()
        image_path = os.path.join(output_dir, f"spec_{i:04d}.png")
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"Saved: {image_path}")

# Run
if __name__ == "__main__":
    audio_to_rgb_spectrogram("jazz_dataset.wav", "riffusion_specs")