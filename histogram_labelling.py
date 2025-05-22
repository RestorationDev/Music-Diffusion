
import os
import cv2
import numpy as np

input_dir = 'spec_images'
output_dir = 'spec_images_matched'
reference_image_path = 'reference.png'  # will replace later

os.makedirs(output_dir, exist_ok=True)

reference = cv2.imread(reference_image_path)

def match_histograms(src, ref):
    matched = np.zeros_like(src)
    for i in range(src.shape[2]):
        src_hist, bins = np.histogram(src[..., i].flatten(), 256, [0, 256])
        ref_hist, bins = np.histogram(ref[..., i].flatten(), 256, [0, 256])

        src_cdf = np.cumsum(src_hist).astype(np.float64)
        src_cdf /= src_cdf[-1]
        ref_cdf = np.cumsum(ref_hist).astype(np.float64)
        ref_cdf /= ref_cdf[-1]

        interp_values = np.interp(src_cdf, ref_cdf, bins[:-1])
        matched[..., i] = np.interp(src[..., i].flatten(), bins[:-1], interp_values).reshape(src[..., i].shape)

    return matched

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        input_path = os.path.join(input_dir, filename)
        image = cv2.imread(input_path)

        matched = match_histograms(image, reference)

        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, np.clip(matched, 0, 255).astype(np.uint8))