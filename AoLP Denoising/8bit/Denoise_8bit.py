import cv2
import numpy as np
import os
from pathlib import Path
import bm3d


def denoise_with_bm3d(scene_name: str, base_dir: str = ".", sigma_psd: float = 0.1):
    """
    Loads pre-calculated P1 and P2 images, denoises them using BM3D,
    and saves the resulting denoised AoLP images.

    """
    print(f"--- Denoising scene with BM3D: {scene_name} ---")
    print(f"Using BM3D noise sigma: {sigma_psd}")

    # --- 1. Setup Paths ---
    input_dir = Path(base_dir) / "output" / scene_name
    output_dir = input_dir

    if not input_dir.exists():
        print(f"Error: Input directory not found at {input_dir}")
        print("Please run the previous calculation script first.")
        return

    # --- 2. Read P1 and P2 images and normalize to [0, 1] float ---
    print("Reading and normalizing P1 and P2 images...")
    p1_path = input_dir / "P1.png"
    p2_path = input_dir / "P2.png"

    if not p1_path.exists() or not p2_path.exists():
        print(f"Error: P1.png or P2.png not found in {input_dir}")
        return

    # Read as 8-bit grayscale
    p1_8bit = cv2.imread(str(p1_path), cv2.IMREAD_UNCHANGED)
    p2_8bit = cv2.imread(str(p2_path), cv2.IMREAD_UNCHANGED)

    # Convert to float and normalize
    p1_float_norm = p1_8bit.astype(np.float32) / 255.0
    p2_float_norm = p2_8bit.astype(np.float32) / 255.0

    # --- 3. Denoise with BM3D ---
    print("Applying BM3D to P1...")
    p1_denoised_norm = bm3d.bm3d(p1_float_norm, sigma_psd=sigma_psd)

    print("Applying BM3D to P2...")
    p2_denoised_norm = bm3d.bm3d(p2_float_norm, sigma_psd=sigma_psd)

    # --- 4. Calculate Denoised AoLP ---
    print("Calculating new AoLP from denoised precursors...")

    # Convert the denoised [0, 1] float images back to their physical [-1, 1] range
    p1_denoised_physical = p1_denoised_norm * 2.0 - 1.0
    p2_denoised_physical = p2_denoised_norm * 2.0 - 1.0

    # Calculate the new AoLP
    AoLP_denoised = 0.5 * np.arctan2(p2_denoised_physical, p1_denoised_physical)

    # --- 5. Save Denoised Images ---
    print("Scaling and saving denoised images...")

    def save_8bit_image(filename, data):
        """Helper function to save float data as an 8-bit PNG."""
        clipped_data = np.clip(data, 0, 255)
        cv2.imwrite(str(output_dir / filename), clipped_data.astype(np.uint8))

    # Save denoised P1 and P2 (scaled back to 8-bit range)
    p1_denoised_8bit = p1_denoised_norm * 255.0
    p2_denoised_8bit = p2_denoised_norm * 255.0
    save_8bit_image("P1_denoised_bm3d.png", p1_denoised_8bit)
    save_8bit_image("P2_denoised_bm3d.png", p2_denoised_8bit)

    # Save denoised AoLP (grayscale 8-bit)
    AoLP_denoised_save = ((AoLP_denoised + np.pi / 2) / np.pi) * 255.0
    save_8bit_image("AoLP_denoised_bm3d.png", AoLP_denoised_save)

    # --- 6. Generate and Save HSV Visualization for Denoised AoLP ---
    print("Generating HSV visualization for denoised AoLP...")

    H_float = ((AoLP_denoised + np.pi / 2) / np.pi) * 360.0
    H_float = H_float.astype(np.float32)
    S_float = np.ones(H_float.shape, dtype=np.float32)
    V_float = np.ones(H_float.shape, dtype=np.float32)

    hsv_image_float = cv2.merge([H_float, S_float, V_float])
    bgr_image_float = cv2.cvtColor(hsv_image_float, cv2.COLOR_HSV2BGR)
    bgr_image_8bit = bgr_image_float * 255.0

    save_8bit_image("AoLP_denoised_bm3d_color.png", bgr_image_8bit)

    print(f"--- BM3D denoising for {scene_name} complete. ---")


if __name__ == '__main__':

    scene_name = "scene_1"
    sigma_value = 0.005

    denoise_with_bm3d(scene_name, base_dir=".", sigma_psd=sigma_value)
