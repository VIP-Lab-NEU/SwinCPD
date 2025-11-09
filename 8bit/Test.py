import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
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
    """Saves a 3-channel tensor [0,106] as a PIL image."""
    os.makedirs(output_dir, exist_ok=True)
    img_tensor_rgb = torch.clamp(tensor_3ch.squeeze(0).cpu(), 0, 1)
    img = transforms.ToPILImage()(img_tensor_rgb)
    try:
        img.save(os.path.join(output_dir, filename))
    except Exception as e:
        print(f"Error saving output image {filename} to {output_dir}: {e}")


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

        if not self.valid_scenes:
            raise FileNotFoundError(f"No matching LR/HR scenes found for split {split}")

        self.to_tensor = transforms.ToTensor()

    def _get_scene_ids(self):
        lr_split_path = os.path.join(self.lr_base_path, self.split)
        if not os.path.isdir(lr_split_path): return []
        scene_ids_numeric = []
        for d in os.listdir(lr_split_path):
            full_path = os.path.join(lr_split_path, d)
            if os.path.isdir(full_path) and d.isdigit():
                scene_ids_numeric.append(d)
        return sorted(scene_ids_numeric, key=int)

    def _group_paths(self, base_path, prefix, suffix=".png", is_hr=False):
        grouped_paths = {}
        split_path = os.path.join(base_path, self.split)
        for scene_id in self.scene_ids:
            scene_paths = {}
            scene_input_folder = os.path.join(split_path, scene_id)
            valid_scene = True
            for angle in self.angles:
                if is_hr:
                    f_path = os.path.join(scene_input_folder, f'gt_{angle}', f"{scene_id}{suffix}")
                else:
                     f_path = os.path.join(scene_input_folder, f"{prefix}{angle}{suffix}")
                if os.path.exists(f_path):
                    scene_paths[angle] = f_path
                else:
                    valid_scene = False; break
            if valid_scene:
                grouped_paths[scene_id] = scene_paths
        return grouped_paths

    def __len__(self):
        return len(self.valid_scenes)

    def __getitem__(self, idx):
        scene_id = self.valid_scenes[idx]
        lr_paths = self.lr_image_paths[scene_id]
        hr_paths = self.hr_image_paths[scene_id]

        try:
            lr_images = [Image.open(lr_paths[angle]).convert('RGB') for angle in self.angles]
            hr_images = [Image.open(hr_paths[angle]).convert('RGB') for angle in self.angles]
        except Exception as e:
             print(f"Error loading images for scene {scene_id}: {e}")
             return None, None, scene_id

        lr_tensors = [self.to_tensor(img) for img in lr_images]
        hr_tensors = [self.to_tensor(img) for img in hr_images]
        lr_stacked = torch.cat(lr_tensors, dim=0)
        hr_stacked = torch.cat(hr_tensors, dim=0)

        return lr_stacked, hr_stacked, scene_id


# --- Main Testing Script ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # --- Load Model ---
    if not os.path.exists(MODEL_CHECKPOINT_PATH):
        print(f"Error: Model checkpoint not found at {MODEL_CHECKPOINT_PATH}")
        sys.exit(1)

    # Instantiate the model structure
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
        model.load_state_dict(checkpoint_data['model_state_dict'])
        print(f"Model weights loaded successfully from epoch {checkpoint_data.get('epoch', 'N/A')}.")
    except KeyError:
        print("Warning: Loading state dict directly (expected checkpoint format not found).")
        try:
            model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE))
            print("Model weights loaded successfully (direct state_dict).")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("Ensure the model definition matches the saved weights.")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Ensure the model definition matches the saved weights.")
        sys.exit(1)

    model.eval()

    # --- Load Test Data ---
    print(f"Loading test data from LR: {LR_DATA_PATH}, HR: {HR_DATA_PATH}")
    try:
        test_dataset = PolarizationTestDataset(LR_DATA_PATH, HR_DATA_PATH, TEST_SPLIT, SCALE_FACTOR)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        print(f"Found {len(test_dataset)} scenes in the test set.")
    except FileNotFoundError as e:
        print(f"Error initializing dataset: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error creating DataLoader: {e}")
        sys.exit(1)

    # --- Testing Loop ---
    print(f"\nStarting testing on split: {TEST_SPLIT}")
    pbar_test = tqdm(test_loader, desc=f"Testing Scenes")

    for lr_img, hr_img, scene_id_list in pbar_test:
        scene_id = scene_id_list[0]
        if lr_img is None or hr_img is None:
             print(f"Skipping scene {scene_id} due to loading error.")
             continue

        lr_img, hr_img = lr_img.to(DEVICE), hr_img.to(DEVICE)

        # --- Inference ---
        with torch.no_grad():
            sr_img = model(lr_img)
            
        sr_img = torch.clamp(sr_img, 0.0, 1.0)
        scene_sr_output_dir = os.path.join(OUTPUT_PATH, TEST_SPLIT, scene_id)
        os.makedirs(scene_sr_output_dir, exist_ok=True)
        for i, angle in enumerate(test_dataset.angles):
            sr_rgb_img = sr_img[:, i * 3:(i + 1) * 3, :, :]
            output_filename = f"Restored_{angle}.png"
            save_output_image(sr_rgb_img, output_filename, scene_sr_output_dir)

    print("Testing finished")
