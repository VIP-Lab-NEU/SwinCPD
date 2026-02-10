import cv2
import numpy as np
import os
from pathlib import Path
import bm3d


def unsharp_mask_stokes(p_channel, sigma=1.0, strength=1.0):
    """
    Applies Unsharp Masking to a Stokes parameter channel (P1 or P2).

    Args:
        p_channel: The normalized Stokes parameter (2D numpy array, float).
        sigma: Gaussian blur radius.
        strength: The magnitude of the sharpening (lambda).

    """
    # 1. Create the Low-Pass (blurred) version
    blurred = cv2.GaussianBlur(p_channel, ksize=(0, 0), sigmaX=sigma)

    # 2. Create the High-Pass mask
    high_pass_mask = p_channel - blurred

    # 3. Add details back to the original image
    sharpened = p_channel + (strength * high_pass_mask)

    # 4. Clip to physically valid range
    sharpened = np.clip(sharpened, -1.0, 1.0)

    return sharpened

def denoise_with_bm3d(scene_name: str, base_dir: str = ".", sigma_psd: float = 0.1):
    """
    Loads pre-calculated P1 and P2 images, denoises them using BM3D,
    and saves the resulting denoised AoLP and DoLP images.

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

    # Read as 16-bit grayscale
    p1_16bit = cv2.imread(str(p1_path), cv2.IMREAD_UNCHANGED)
    p2_16bit = cv2.imread(str(p2_path), cv2.IMREAD_UNCHANGED)

    # Convert to float and normalize
    p1_float_norm = p1_16bit.astype(np.float32) / 65535.0
    p2_float_norm = p2_16bit.astype(np.float32) / 65535.0

    # --- 3. Denoise with BM3D ---
    print("Applying BM3D to P1...")
    p1_denoised_norm = bm3d.bm3d(p1_float_norm, sigma_psd=sigma_psd)

    print("Applying BM3D to P2...")
    p2_denoised_norm = bm3d.bm3d(p2_float_norm, sigma_psd=sigma_psd)

    # --- 4. Calculate Denoised AoLP, DoLP, and Enhanced DoLP ---
    print("Calculating new AoLP and DoLP from denoised precursors...")

    # Convert the denoised [0, 1] float images back to their physical [-1, 1] range
    p1_denoised_physical = p1_denoised_norm * 2.0 - 1.0
    p2_denoised_physical = p2_denoised_norm * 2.0 - 1.0

    # (Optional) Enhance P1 and P2 to calculate more accurate DoLP
    p1_enhanced = unsharp_mask_stokes(p1_denoised_physical, sigma=σ, strength=λ)
    p2_enhanced = unsharp_mask_stokes(p2_denoised_physical, sigma=σ, strength=λ)

    # Calculate the new AoLP and DoLP
    AoLP_denoised = 0.5 * np.arctan2(p2_denoised_physical, p1_denoised_physical)
    DoLP_denoised = np.sqrt(p1_denoised_physical ** 2 + p2_denoised_physical ** 2)    # Optional
    DoLP_enhanced = np.sqrt(p1_enhanced ** 2 + p2_enhanced ** 2)                      # Optional

    # --- 5. Save Denoised Images ---
    print("Scaling and saving denoised images...")

    def save_16bit_image(filename, data):
        """Helper function to save float data as a 16-bit PNG."""
        clipped_data = np.clip(data, 0, 65535)
        cv2.imwrite(str(output_dir / filename), clipped_data.astype(np.uint16))

    # Save denoised P1 and P2 (scaled back to 16-bit range)
    p1_denoised_16bit = p1_denoised_norm * 65535.0
    p2_denoised_16bit = p2_denoised_norm * 65535.0
    save_16bit_image("P1_denoised_bm3d.png", p1_denoised_16bit)
    save_16bit_image("P2_denoised_bm3d.png", p2_denoised_16bit)

    # Save denoised AoLP (grayscale 16-bit)
    AoLP_denoised_save = ((AoLP_denoised + np.pi / 2) / np.pi) * 65535.0
    save_16bit_image("AoLP_denoised_bm3d.png", AoLP_denoised_save)

    # Save denoised DoLP (grayscale 16-bit)
    DoLP_denoised_save = DoLP_denoised * 65535.0
    save_16bit_image("DoLP_denoised_bm3d.png", DoLP_denoised_save)

    # Save enhanced DoLP (grayscale 16-bit)
    DoLP_enhanced_save = DoLP_enhanced * 65535.0
    save_16bit_image("DoLP_enhanced.png", DoLP_enhanced_save)

    # --- 6. Generate and Save HSV Visualization for Denoised AoLP ---
    print("Generating HSV visualization for denoised AoLP...")

    H_float = ((AoLP_denoised + np.pi / 2) / np.pi) * 360.0
    H_float = H_float.astype(np.float32)
    S_float = np.ones(H_float.shape, dtype=np.float32)
    V_float = np.ones(H_float.shape, dtype=np.float32)

    hsv_image_float = cv2.merge([H_float, S_float, V_float])
    bgr_image_float = cv2.cvtColor(hsv_image_float, cv2.COLOR_HSV2BGR)
    bgr_image_16bit = bgr_image_float * 65535.0

    save_16bit_image("AoLP_denoised_bm3d_color.png", bgr_image_16bit)

    print(f"--- BM3D denoising for {scene_name} complete. ---")


if __name__ == '__main__':

    scene_name = "scene_1"
    sigma_value = 0.005

    σ = 1.0
    λ = 0.2

    denoise_with_bm3d(scene_name, base_dir=".", sigma_psd=sigma_value)
