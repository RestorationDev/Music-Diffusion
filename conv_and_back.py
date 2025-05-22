import librosa
import numpy as np
import os
import soundfile as sf

# ---------- Configuration ----------
input_wav = "jazz_dataset.wav"
clip_duration = 5          # seconds
sr = 22050                 # Sample rate
n_fft = 2048
hop_length = 512
n_mels = 128
fmax = 8000
n_iter = 100               # Griffin-Lim iterations
output_spec_dir = "spec_npy"
output_audio_dir = "reconstructed_audio"

# ---------- Setup ----------
os.makedirs(output_spec_dir, exist_ok=True)
os.makedirs(output_audio_dir, exist_ok=True)

# ---------- Load Audio ----------
y, sr = librosa.load(input_wav, sr=sr)
samples_per_clip = clip_duration * sr
total_clips = len(y) // samples_per_clip

# ---------- Loop: Convert to Mel Spectrogram and Save ----------
for i in range(total_clips):
    start = i * samples_per_clip
    end = start + samples_per_clip
    clip = y[start:end]

    # Mel spectrogram -> dB
    S = librosa.feature.melspectrogram(y=clip, sr=sr, n_fft=n_fft,
                                       hop_length=hop_length, n_mels=n_mels, fmax=fmax)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Save as .npy
    spec_path = os.path.join(output_spec_dir, f"spec_{i:04d}.npy")
    np.save(spec_path, S_dB)

    # Reconstruct audio
    S_recon = librosa.db_to_power(S_dB)
    y_recon = librosa.feature.inverse.mel_to_audio(S_recon, sr=sr,
                                                   n_fft=n_fft,
                                                   hop_length=hop_length,
                                                   n_iter=n_iter)
    audio_path = os.path.join(output_audio_dir, f"recon_{i:04d}.wav")
    sf.write(audio_path, y_recon, sr)
    print(f"Processed segment {i}: Saved spec and audio.")

print("✅ All segments processed.")