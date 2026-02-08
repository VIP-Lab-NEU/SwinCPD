import cv2
import numpy as np
import os
from pathlib import Path


def process_polarization_scene(folder_path: str, base_dir: str = "."):
    """
    Reads four 16-bit RGB polarization images (0, 45, 90, 135 degrees),
    calculates key polarization parameters (Stokes, DoLP, AoLP), and saves
    them as 16-bit PNG files.

    """
    print(f"--- Processing scene: {folder_path} ---")

    # --- 1. Setup Paths and Directories ---
    input_dir = Path(base_dir) / folder_path
    output_dir = Path(base_dir) / "output" / folder_path

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {output_dir.resolve()}")

    angles = [0, 45, 90, 135]
    intensities = {}

    # --- 2. Read Images and Convert to Luminance ---
    print("Reading and converting images to luminance...")
    for angle in angles:
        img_path = input_dir / f"{angle}.PNG"
        if not img_path.exists():
            print(f"Error: Input file not found at {img_path}")
            return

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

        if img_bgr is None:
            print(f"Error: Failed to read image at {img_path}")
            return

        if img_bgr.dtype != np.uint16:
            print(f"Warning: Image {img_path} is not 16-bit. It will be converted.")
            img_bgr = img_bgr.astype(np.uint16)

        img_bgr_float = img_bgr.astype(np.float32)

        luminance = (0.114 * img_bgr_float[:, :, 0] +
                     0.587 * img_bgr_float[:, :, 1] +
                     0.299 * img_bgr_float[:, :, 2])

        intensities[angle] = luminance

    # --- 3. Calculate Stokes Parameters and Polarization Attributes ---
    print("Calculating Stokes parameters and polarization attributes...")
    I0, I45, I90, I135 = intensities[0], intensities[45], intensities[90], intensities[135]

    epsilon = 1e-8

    S0 = 0.5*(I0 + I45 + I90 + I135)
    S1 = I0 - I90
    S2 = I45 - I135

    P1 = S1 / (S0 + epsilon)
    P2 = S2 / (S0 + epsilon)

    DoLP = np.sqrt(S1 ** 2 + S2 ** 2) / (S0 + epsilon)
    DoLP = np.clip(DoLP, 0, 1)

    AoLP = 0.5 * np.arctan2(S2, S1)

    # --- 4. Scale and Save 16-bit Images ---
    print("Scaling and saving output images...")

    def save_16bit_image(filename, data):
        """Helper function to save float data as a 16-bit PNG."""
        clipped_data = np.clip(data, 0, 65535)
        cv2.imwrite(str(output_dir / filename), clipped_data.astype(np.uint16))

    S0_save = S0 / 2.0
    save_16bit_image("S0.png", S0_save)

    S1_save = (S1 + 65535.0) / 2.0
    S2_save = (S2 + 65535.0) / 2.0
    save_16bit_image("S1.png", S1_save)
    save_16bit_image("S2.png", S2_save)

    P1_save = (P1 + 1.0) * 0.5 * 65535.0
    P2_save = (P2 + 1.0) * 0.5 * 65535.0
    save_16bit_image("P1.png", P1_save)
    save_16bit_image("P2.png", P2_save)

    DoLP_save = DoLP * 65535.0
    save_16bit_image("DoLP.png", DoLP_save)

    AoLP_save = ((AoLP + np.pi / 2) / np.pi) * 65535.0
    save_16bit_image("AoLP.png", AoLP_save)

    # --- 5. Generate and Save HSV Visualization for AoLP ---
    print("Generating HSV visualization for AoLP...")

    # Create float32 arrays for HSV channels in the ranges OpenCV expects for floats

    # Hue (H): Map AoLP from [-pi/2, pi/2] to OpenCV's float Hue range [0, 360]
    H_float = ((AoLP + np.pi / 2) / np.pi) * 360.0
    H_float = H_float.astype(np.float32)

    # Saturation (S): Full saturation, represented as 1.0 for float
    S_float = np.ones(H_float.shape, dtype=np.float32)

    # Value (V): Full brightness, represented as 1.0 for float
    V_float = np.ones(H_float.shape, dtype=np.float32)

    # Merge float channels into a single 3-channel HSV image
    hsv_image_float = cv2.merge([H_float, S_float, V_float])

    # Convert from HSV to BGR color space
    bgr_image_float = cv2.cvtColor(hsv_image_float, cv2.COLOR_HSV2BGR)

    # Scale the BGR float image [0, 1] back to the 16-bit integer range [0, 65535]
    bgr_image_16bit = bgr_image_float * 65535.0

    # Save the final 16-bit color image
    save_16bit_image("AoLP_color.png", bgr_image_16bit)

    print(f"--- Processing for {folder_path} complete. ---")


if __name__ == '__main__':

    folder_path = "scene_1"  # <--- Change this to your actual name of the folder that contains the I0 I45 I90 I135 images
    process_polarization_scene(folder_path, base_dir=".")
