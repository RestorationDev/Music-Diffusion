import numpy as np
import soundfile as sf
from PIL import Image
import librosa
import matplotlib.cm as cm
from matplotlib import colormaps

def rgb_spectrogram_to_audio(image_path, output_wav_path, hop_length=256, sr=44100, n_iter=64):
    # Load image and normalize
    img = Image.open(image_path).convert("RGB")
    img = np.array(img).astype(np.float32) / 255.0

    # Invert viridis colormap using nearest-neighbor LUT
    cmap = colormaps['viridis']
    lut = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    img_flat = (img.reshape(-1, 3) * 255).astype(np.uint8)

    # Match each RGB pixel to the closest colormap entry
    grayscale = np.array([
        np.argmin(np.linalg.norm(lut - rgb, axis=1))
        for rgb in img_flat
    ]).reshape(img.shape[:2])

    # Recover magnitude spectrogram from grayscale
    S_db_norm = grayscale / 255.0
    S_db = S_db_norm * 80.0 - 80.0
    S_mag = librosa.db_to_amplitude(S_db)

    # Dynamically infer correct n_fft from spectrogram shape
    n_fft = (S_mag.shape[0] - 1) * 2

    # Griffin-Lim reconstruction
    y = librosa.griffinlim(S_mag, n_iter=n_iter, hop_length=hop_length, n_fft=n_fft)
    sf.write(output_wav_path, y, sr)
    print(f"✅ Saved reconstructed audio: {output_wav_path}")

# Run
if __name__ == "__main__":
    input_img = "riffusion_specs/spec_0175.png"
    output_wav = "reconstructed_0000.wav"
    rgb_spectrogram_to_audio(input_img, output_wav)