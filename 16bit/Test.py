import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tqdm import tqdm
import sys

from SwinCPD_Model import SwinCPD as model

# --- Configuration ---
LR_DATA_PATH = 'Processed_Dataset'
HR_DATA_PATH = 'Dataset'
MODEL_CHECKPOINT_PATH = 'Models/SwinCPD.pth' 
OUTPUT_PATH = 'Output'
TEST_SPLIT = 'Test'
SCALE_FACTOR = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
    
def save_output_image(tensor_3ch, filename, output_dir):
    """Saves a 3-channel tensor [0,1] as a 16-bit BGR PNG using OpenCV."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        img_tensor_rgb = torch.clamp(tensor_3ch.squeeze(0).cpu().detach(), 0, 1); img_np_hwc = img_tensor_rgb.permute(1, 2, 0).numpy()
        img_np_16bit = (img_np_hwc * 65535.0).clip(0, 65535).astype(np.uint16); img_np_bgr = cv2.cvtColor(img_np_16bit, cv2.COLOR_RGB2BGR)
        output_path = os.path.join(output_dir, filename); success = cv2.imwrite(output_path, img_np_bgr)
        if not success: print(f"Warning: cv2.imwrite failed {output_path}")
    except Exception as e: print(f"Error saving output img {filename}: {e}")

# --- Dataset Definition ---
class PolarizationTestDataset(Dataset):
    def __init__(self, lr_base_path, hr_base_path, split, scale_factor):
        self.lr_base_path = lr_base_path
        self.hr_base_path = hr_base_path
        self.split = split
        self.scale_factor = scale_factor
        self.angles = [0, 45, 90, 135]
        self.scene_ids = self._get_scene_ids()
        self.lr_image_paths = self._group_paths(lr_base_path, prefix="Refine_Input_")
        self.hr_image_paths = self._group_paths(hr_base_path, prefix="", suffix=".png", is_hr=True)
        self.valid_scenes = sorted(list(self.lr_image_paths.keys() & self.hr_image_paths.keys()), key=int)
        if not self.valid_scenes: raise FileNotFoundError(f"No matching LR/HR scenes found for test split {split}")

    def _get_scene_ids(self):
        scene_ids = []; lr_split_path = os.path.join(self.lr_base_path, self.split)
        if os.path.isdir(lr_split_path):
            try: scene_ids_str = sorted([d for d in os.listdir(lr_split_path) if os.path.isdir(os.path.join(lr_split_path, d)) and d.isdigit()], key=int); scene_ids = [str(sid) for sid in scene_ids_str]
            except ValueError: scene_ids = sorted([d for d in os.listdir(lr_split_path) if os.path.isdir(os.path.join(lr_split_path, d))])
        return scene_ids

    def _group_paths(self, base_path, prefix, suffix=".png", is_hr=False):
        grouped_paths = {}; split_path = os.path.join(base_path, self.split)
        for scene_id in self.scene_ids:
            scene_paths = {}; valid_scene = True; scene_input_folder = os.path.join(split_path, scene_id)
            for angle in self.angles:
                 if is_hr: hr_gt_folder = os.path.join(base_path, self.split, scene_id, f'gt_{angle}')
                 else: lr_folder = os.path.join(base_path, self.split, scene_id)

                 fname_lower = f"{scene_id}{suffix}" if is_hr else f"{prefix}{angle}{suffix}"
                 fname_upper = fname_lower.replace(".png",".PNG")

                 path_lower = os.path.join(hr_gt_folder, fname_lower) if is_hr else os.path.join(lr_folder, fname_lower)
                 path_upper = os.path.join(hr_gt_folder, fname_upper) if is_hr else os.path.join(lr_folder, fname_upper)

                 f_path = None
                 if os.path.exists(path_lower): f_path = path_lower
                 elif os.path.exists(path_upper): f_path = path_upper

                 if f_path: scene_paths[angle] = f_path
                 else: valid_scene = False; break
            if valid_scene: grouped_paths[scene_id] = scene_paths
        return grouped_paths

    def __len__(self): return len(self.valid_scenes)

    def __getitem__(self, idx): # Loads 16bit, normalizes, stacks to 12ch tensor
        scene_id = self.valid_scenes[idx]; lr_paths = self.lr_image_paths[scene_id]; hr_paths = self.hr_image_paths[scene_id]
        try:
            lr_images_bgr = [cv2.imread(lr_paths[angle], cv2.IMREAD_UNCHANGED) for angle in self.angles];
            hr_images_bgr = [cv2.imread(hr_paths[angle], cv2.IMREAD_UNCHANGED) for angle in self.angles]
            if any(img is None for img in lr_images_bgr) or any(img is None for img in hr_images_bgr): raise IOError("Load fail")
            if any(img.dtype != np.uint16 for img in lr_images_bgr) or any(img.dtype != np.uint16 for img in hr_images_bgr): raise ValueError("Not uint16")
            if any(len(img.shape)!=3 or img.shape[2]!=3 for img in lr_images_bgr) or any(len(img.shape)!=3 or img.shape[2]!=3 for img in hr_images_bgr): raise ValueError("Not 3ch BGR")
            lr_images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in lr_images_bgr]; hr_images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in hr_images_bgr]
            lr_np_stacked = np.stack(lr_images_rgb, axis=0); hr_np_stacked = np.stack(hr_images_rgb, axis=0)
            lr_h, lr_w = lr_np_stacked.shape[1:3];
            hr_h, hr_w = hr_np_stacked.shape[1:3]
            lr_np_12ch = lr_np_stacked.transpose(0, 3, 1, 2).reshape(12, lr_h, lr_w);
            hr_np_12ch = hr_np_stacked.transpose(0, 3, 1, 2).reshape(12, hr_h, hr_w)
            lr_norm = lr_np_12ch.astype(np.float32) / 65535.0;
            hr_norm = hr_np_12ch.astype(np.float32) / 65535.0
            lr_tensor = torch.from_numpy(lr_norm);
            hr_tensor = torch.from_numpy(hr_norm)
            return lr_tensor, hr_tensor, scene_id
        except Exception as e: print(f"Error test scene {scene_id}: {e}"); return None, None, scene_id

# --- Main Testing Script ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # --- Load Model ---
    print(f"Loading model checkpoint...")
    if not os.path.exists(MODEL_CHECKPOINT_PATH): print(f"Error: Model checkpoint not found"); sys.exit(1)
    model = model(
        img_size=96, 
        in_chans=12, 
        out_chans=12, 
        embed_dim=48,
        depths=[6, 6, 6, 6, 6, 6], 
        num_heads=[6, 6, 6, 6, 6, 6], 
        window_size=8, 
        mlp_ratio=2.
    ).to(DEVICE)
    try: 
        checkpoint_data = torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE)
        state_dict_key = 'model_state_dict' if 'model_state_dict' in checkpoint_data else None
        if state_dict_key: model.load_state_dict(checkpoint_data[state_dict_key]); print(f"Loaded weights from epoch {checkpoint_data.get('epoch', 'N/A')}.")
        else: model.load_state_dict(checkpoint_data); print("Loaded weights (direct state_dict).") # Fallback
    except Exception as e: print(f"Error loading model weights: {e}"); sys.exit(1)
    model.eval()

    # --- Load Test Data ---
    print(f"Loading test data...")
    try: test_dataset = PolarizationTestDataset(LR_DATA_PATH, HR_DATA_PATH, TEST_SPLIT, SCALE_FACTOR); test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2); print(f"Found {len(test_dataset)} scenes.")
    except Exception as e: print(f"Error loading dataset: {e}"); sys.exit(1)

    # --- Testing Loop ---
    print(f"\nStarting testing...")
    pbar_test = tqdm(test_loader, desc=f"Testing Scenes")
    for lr_img, hr_img, scene_id_list in pbar_test:
        scene_id = scene_id_list[0] # B=1
        if lr_img is None: print(f"Skipping {scene_id} (load error)."); continue
        lr_img, hr_img = lr_img.to(DEVICE), hr_img.to(DEVICE)
        with torch.no_grad(): sr_img = model(lr_img)
        sr_img = torch.clamp(sr_img, 0.0, 1.0)

        # --- Save Images ---
        scene_sr_output_dir = os.path.join(OUTPUT_PATH, TEST_SPLIT, scene_id)
        for i, angle in enumerate(test_dataset.angles):
            sr_rgb_img = sr_img[:, i * 3:(i + 1) * 3, :, :]
            output_filename = f"Restored_{angle}.png"
            save_output_image(sr_rgb_img, output_filename, scene_sr_output_dir)
            
    print("Testing finished")
