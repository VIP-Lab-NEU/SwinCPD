import os
import numpy as np
import cv2
from tqdm import tqdm
import sys

# --- Import the interpolation tool ---
from scipy.interpolate import NearestNDInterpolator

# --- Configuration ---
BASE_DATASET_PATH = 'Dataset'
OUTPUT_BASE_PATH = 'Processed_Dataset'
SPLITS = {
    'Train': (1, 30),
    'Val': (31, 32),
    'Test': (33, 40)    # Match your dataset allocation
}
SENSOR_LAYOUT_POLARIZATION = {
    (0, 0): 90, (0, 1): 45, (0, 2): 90, (0, 3): 45,
    (1, 0): 135, (1, 1): 0, (1, 2): 135, (1, 3): 0,
    (2, 0): 90, (2, 1): 45, (2, 2): 90, (2, 3): 45,
    (3, 0): 135, (3, 1): 0, (3, 2): 135, (3, 3): 0,
}
SENSOR_LAYOUT_COLOR = {
    (0, 0): 'R', (0, 1): 'R', (0, 2): 'G', (0, 3): 'G',
    (1, 0): 'R', (1, 1): 'R', (1, 2): 'G', (1, 3): 'G',
    (2, 0): 'G', (2, 1): 'G', (2, 2): 'B', (2, 3): 'B',
    (3, 0): 'G', (3, 1): 'G', (3, 2): 'B', (3, 3): 'B',
}
POLARIZATION_ANGLES = [0, 45, 90, 135]
COLORS = ['R', 'G', 'B']


def fill_channel_nn(sparse_channel):

    H, W = sparse_channel.shape
    dtype = sparse_channel.dtype

    # Find the coordinates and values of all known (non-zero) pixels
    mask = sparse_channel != 0
    if not mask.any():
        # If a channel is genuinely empty (e.g., a scene with no blue), return a black image
        return sparse_channel

    rows, cols = np.where(mask)
    values = sparse_channel[mask]

    # Create the grid of all pixel coordinates in the image to be queried
    grid_rows, grid_cols = np.mgrid[0:H, 0:W]

    # Build the nearest-neighbor interpolator function
    interp_func = NearestNDInterpolator(list(zip(rows, cols)), values)

    # Call the interpolator on the full grid of coordinates and cast to original dtype
    return interp_func(grid_rows, grid_cols).astype(dtype)


# --- Main Processing ---
print(f"Input dataset root: {BASE_DATASET_PATH}")
print(f"Output directory: {OUTPUT_BASE_PATH}")

for split_name, (start_id, end_id) in SPLITS.items():
    print(f"\nProcessing split: {split_name} (Scenes {start_id}-{end_id})")
    split_input_path = os.path.join(BASE_DATASET_PATH, split_name)
    split_output_path = os.path.join(OUTPUT_BASE_PATH, split_name)
    if not os.path.isdir(split_input_path):
        print(f"Warning: Input directory for split '{split_name}' not found. Skipping.")
        continue

    try:
        scene_ids_str = sorted([d for d in os.listdir(split_input_path) if
                                os.path.isdir(os.path.join(split_input_path, d)) and d.isdigit()], key=int)
        scene_ids = [int(sid) for sid in scene_ids_str if start_id <= int(sid) <= end_id]
        if not scene_ids:
            print(f"Warning: No valid scene IDs found in range {start_id}-{end_id} for split {split_name}.")
            continue
    except Exception as e:
        print(f"Error listing/sorting scene IDs for split {split_name}: {e}. Skipping split.")
        continue

    for scene_id in tqdm(scene_ids, desc=f"  Processing {split_name}"):
        scene_str = str(scene_id)
        scene_input_folder = os.path.join(split_input_path, scene_str)
        scene_output_folder = os.path.join(split_output_path, scene_str)

        raw_input_path = os.path.join(scene_input_folder, 'net_input', f"{scene_str}.png")
        if not os.path.exists(raw_input_path):
            raw_input_path = os.path.join(scene_input_folder, 'net_input', f"{scene_str}.PNG")

        if not os.path.exists(raw_input_path):
            print(f"Warning: Raw input not found for Scene {scene_str}. Skipping.")
            continue

        os.makedirs(scene_output_folder, exist_ok=True)

        try:
            # --- Load Raw Image ---
            raw_img_unchanged = cv2.imread(raw_input_path, cv2.IMREAD_UNCHANGED)
            if raw_img_unchanged is None:
                print(f"Warning: OpenCV failed to load {raw_input_path}. Skipping.")
                continue

            raw_data = cv2.cvtColor(raw_img_unchanged, cv2.COLOR_BGR2GRAY) if len(
                raw_img_unchanged.shape) == 3 else raw_img_unchanged
            if np.max(raw_data) == 0:
                print(f"Warning: Raw input for scene {scene_str} is all black. Skipping.")
                continue

            # --- Create and Populate 12 sparse channels ---
            sparse_channels = {f'{c}_{a}': np.zeros_like(raw_data) for a in POLARIZATION_ANGLES for c in COLORS}
            for r_offset in range(4):
                for c_offset in range(4):
                    angle = SENSOR_LAYOUT_POLARIZATION[(r_offset, c_offset)]
                    color = SENSOR_LAYOUT_COLOR[(r_offset, c_offset)]
                    key = f'{color}_{angle}'
                    sparse_channels[key][r_offset::4, c_offset::4] = raw_data[r_offset::4, c_offset::4]

            # --- Fill each of the 12 channels ---
            filled_channels = {}
            for key, sparse_image in sparse_channels.items():
                filled_channels[key] = fill_channel_nn(sparse_image)

            # --- Stack R, G, B channels for each angle and save ---
            for angle in POLARIZATION_ANGLES:
                R_filled = filled_channels[f'R_{angle}']
                G_filled = filled_channels[f'G_{angle}']
                B_filled = filled_channels[f'B_{angle}']

                demosaiced_rgb = np.stack([R_filled, G_filled, B_filled], axis=-1)

                bgr_image = cv2.cvtColor(demosaiced_rgb, cv2.COLOR_RGB2BGR)
                output_path = os.path.join(scene_output_folder, f'Refine_Input_{angle}.png')
                cv2.imwrite(output_path, bgr_image)

        except Exception as e:
            print(f"\nERROR processing Scene {scene_str}: {e}")
            import traceback

            traceback.print_exc()
            continue

print("\nPreprocessing finished.")


