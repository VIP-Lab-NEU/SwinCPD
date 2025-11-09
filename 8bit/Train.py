import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

# --- Model Selection ---
from SwinCPD_Model import SwinCPD as model

# --- Configuration ---
LR_DATA_PATH = 'Processed_Dataset'
HR_DATA_PATH = 'Dataset'
VALIDATION_OUTPUT_PATH = 'Validation_Outputs'
MODEL_SAVE_PATH = 'Models/SwinCPD.pth'

# --- Augmentation Control ---
USE_AUGMENTATION = True

# --- Loss Weights ---
INTENSITY_WEIGHT = 1.0
Y_WEIGHT = 1.0
C_WEIGHT = 0.1
STOKES_WEIGHT = 1.0

# --- Hyperparameters ---
SCALE_FACTOR = 1
NUM_EPOCHS = 500
BATCH_SIZE = 4
LR_PATCH_SIZE = 96
HR_PATCH_SIZE = LR_PATCH_SIZE
NUM_PATCHES_PER_SCENE_TRAIN = 100
HARD_PATCH_MINING_PERCENTAGE = 1.0    # Optional, default=1.0
LEARNING_RATE = 2e-4
OPTIMIZER_BETAS = (0.9, 0.999)
COSINE_ETA_MIN = 1e-7

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DynamicPatchDataset(Dataset):
    def __init__(self, lr_base_path, hr_base_path, split, scale_factor, lr_patch_size,
                 num_patches_per_scene, mining_percentage, apply_augmentation=False):
        print(f"\nInitializing Dynamic Dataset for split '{split}'...")
        self.lr_base_path = lr_base_path
        self.hr_base_path = hr_base_path
        self.split = split
        self.scale_factor = scale_factor
        self.lr_patch_size = lr_patch_size
        self.hr_patch_size = lr_patch_size * scale_factor
        self.num_patches_per_scene = num_patches_per_scene
        self.mining_percentage = mining_percentage
        self.apply_augmentation = apply_augmentation
        
        self.angles = [0, 45, 90, 135]
        self.lr_patches, self.hr_patches = [], []

        self._find_valid_scenes()
        print(f"  Found {len(self.valid_scenes)} valid scenes for '{split}'.")
        print(f"  On-the-fly augmentation: {'ENABLED' if self.apply_augmentation else 'DISABLED'}")
        print(f"  Hard Patch Mining: Top {int(self.mining_percentage * 100)}% of patches will be used.")

    def _get_scene_ids(self):
        scene_ids = []
        lr_split_path = os.path.join(self.lr_base_path, self.split)
        if os.path.isdir(lr_split_path):
            try:
                scene_ids_str = sorted([d for d in os.listdir(lr_split_path) if os.path.isdir(os.path.join(lr_split_path, d)) and d.isdigit()], key=int)
                scene_ids = [str(sid) for sid in scene_ids_str]
            except ValueError:
                scene_ids = sorted([d for d in os.listdir(lr_split_path) if os.path.isdir(os.path.join(lr_split_path, d))])
        return scene_ids

    def _group_paths(self, base_path, prefix="", suffix=".png", is_hr=False):
        grouped_paths = {}
        for scene_id in self.scene_ids_from_dir:
            scene_paths = {}; valid_scene = True
            for angle in self.angles:
                if is_hr:
                    folder = os.path.join(base_path, self.split, scene_id, f'gt_{angle}')
                    fname_lower = f"{scene_id}{suffix}"
                else:
                    folder = os.path.join(base_path, self.split, scene_id)
                    fname_lower = f"{prefix}{angle}{suffix}"

                fname_upper = fname_lower.replace(".png", ".PNG")
                path_lower = os.path.join(folder, fname_lower)
                path_upper = os.path.join(folder, fname_upper)
                
                f_path = None
                if os.path.exists(path_lower): f_path = path_lower
                elif os.path.exists(path_upper): f_path = path_upper
                
                if f_path:
                    scene_paths[angle] = f_path
                else:
                    valid_scene = False
                    break
            if valid_scene: grouped_paths[scene_id] = scene_paths
        return grouped_paths

    def _find_valid_scenes(self):
        self.scene_ids_from_dir = self._get_scene_ids()
        self.lr_image_paths = self._group_paths(self.lr_base_path, prefix="Refine_Input_")
        self.hr_image_paths = self._group_paths(self.hr_base_path, is_hr=True, prefix="")
        self.valid_scenes = sorted(list(self.lr_image_paths.keys() & self.hr_image_paths.keys()), key=int)
        if not self.valid_scenes:
            raise FileNotFoundError(f"No matching LR/HR scenes found for split {self.split}")

    def generate_epoch_patches(self):
        print("  Generating new set of hard patches for the epoch...")
        candidate_patches = []
        
        for scene_id in tqdm(self.valid_scenes, desc="  Cropping scenes", leave=False):
            try:
                lr_paths = self.lr_image_paths[scene_id]; hr_paths = self.hr_image_paths[scene_id]
                lr_images_bgr = [cv2.imread(lr_paths[angle], cv2.IMREAD_COLOR) for angle in self.angles]
                hr_images_bgr = [cv2.imread(hr_paths[angle], cv2.IMREAD_COLOR) for angle in self.angles]

                if any(img is None for img in lr_images_bgr + hr_images_bgr): continue

                lr_images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in lr_images_bgr]
                hr_images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in hr_images_bgr]
                lr_h, lr_w = lr_images_rgb[0].shape[:2]
                if self.lr_patch_size > lr_h or self.lr_patch_size > lr_w: continue

                for _ in range(self.num_patches_per_scene):
                    lr_x = random.randint(0, lr_w - self.lr_patch_size)
                    lr_y = random.randint(0, lr_h - self.lr_patch_size)
                    hr_x, hr_y = lr_x * self.scale_factor, lr_y * self.scale_factor

                    lr_patches_np = [img[lr_y:lr_y + self.lr_patch_size, lr_x:lr_x + self.lr_patch_size] for img in lr_images_rgb]
                    hr_patches_np = [img[hr_y:hr_y + self.hr_patch_size, hr_x:hr_x + self.hr_patch_size] for img in hr_images_rgb]

                    lr_np_12ch = np.stack(lr_patches_np).transpose(0, 3, 1, 2).reshape(12, self.lr_patch_size, self.lr_patch_size)
                    hr_np_12ch = np.stack(hr_patches_np).transpose(0, 3, 1, 2).reshape(12, self.hr_patch_size, self.hr_patch_size)
                    
                    lr_norm = lr_np_12ch.astype(np.float32) / 255.0
                    hr_norm = hr_np_12ch.astype(np.float32) / 255.0

                    mae = np.mean(np.abs(lr_norm - hr_norm))
                    candidate_patches.append((mae, lr_norm, hr_norm))
            except Exception as e:
                print(f"Warning: Error processing scene {scene_id} for patching: {e}")
        
        if not candidate_patches:
            raise RuntimeError("Failed to generate any patch candidates for the epoch.")
            
        print(f"  Generated {len(candidate_patches)} candidate patches.")
        candidate_patches.sort(key=lambda x: x[0], reverse=True)
        
        num_to_keep = int(len(candidate_patches) * self.mining_percentage)
        hard_patches = candidate_patches[:num_to_keep]
        
        self.lr_patches = np.stack([p[1] for p in hard_patches])
        self.hr_patches = np.stack([p[2] for p in hard_patches])
        
        print(f"  Selected {len(self.lr_patches)} hard patches for training.")

    def __len__(self):
        return len(self.lr_patches)

    def __getitem__(self, idx):
        lr_patch_chw = self.lr_patches[idx].copy()
        hr_patch_chw = self.hr_patches[idx].copy()

        if self.apply_augmentation:
            do_hflip, do_vflip = random.random() < 0.5, random.random() < 0.5
            rot_type = random.randint(0, 2)
            if do_hflip:
                lr_patch_chw, hr_patch_chw = np.flip(lr_patch_chw, axis=2), np.flip(hr_patch_chw, axis=2)
            if do_vflip:
                lr_patch_chw, hr_patch_chw = np.flip(lr_patch_chw, axis=1), np.flip(hr_patch_chw, axis=1)
            if rot_type == 1:
                lr_patch_chw, hr_patch_chw = np.rot90(lr_patch_chw, k=-1, axes=(1, 2)), np.rot90(hr_patch_chw, k=-1, axes=(1, 2))
            elif rot_type == 2:
                lr_patch_chw, hr_patch_chw = np.rot90(lr_patch_chw, k=-2, axes=(1, 2)), np.rot90(hr_patch_chw, k=-2, axes=(1, 2))
            if rot_type == 1:
                permute_indices = np.concatenate((np.arange(6, 9), np.arange(9, 12), np.arange(0, 3), np.arange(3, 6)))
                lr_patch_chw, hr_patch_chw = lr_patch_chw[permute_indices, :, :], hr_patch_chw[permute_indices, :, :]
            
            return torch.from_numpy(np.ascontiguousarray(lr_patch_chw)), torch.from_numpy(np.ascontiguousarray(hr_patch_chw))
        else:
            return torch.from_numpy(lr_patch_chw), torch.from_numpy(hr_patch_chw)

# --- Validation Dataset ---
class PolarizationDatasetVal(Dataset):
    def __init__(self, lr_base_path, hr_base_path, split, scale_factor):
        self.lr_base_path = lr_base_path; self.hr_base_path = hr_base_path; self.split = split; self.scale_factor = scale_factor; self.angles = [0, 45, 90, 135]
        self.scene_ids_from_dir = self._get_scene_ids()
        self.lr_image_paths = self._group_paths(lr_base_path, prefix="Refine_Input_"); self.hr_image_paths = self._group_paths(hr_base_path, is_hr=True, prefix="")
        self.valid_scenes = sorted(list(self.lr_image_paths.keys() & self.hr_image_paths.keys()), key=int)
        if not self.valid_scenes: raise FileNotFoundError(f"No matching LR/HR scenes for val split {split}")
    def _get_scene_ids(self):
        scene_ids = []
        lr_split_path = os.path.join(self.lr_base_path, self.split)
        if os.path.isdir(lr_split_path):
            try:
                scene_ids_str = sorted([d for d in os.listdir(lr_split_path) if os.path.isdir(os.path.join(lr_split_path, d)) and d.isdigit()], key=int)
                scene_ids = [str(sid) for sid in scene_ids_str]
            except ValueError:
                scene_ids = sorted([d for d in os.listdir(lr_split_path) if os.path.isdir(os.path.join(lr_split_path, d))])
        return scene_ids
    def _group_paths(self, base_path, prefix="", suffix=".png", is_hr=False):
        grouped_paths = {}
        for scene_id in self.scene_ids_from_dir:
            scene_paths = {}; valid_scene = True
            for angle in self.angles:
                if is_hr:
                    folder = os.path.join(base_path, self.split, scene_id, f'gt_{angle}')
                    fname_lower = f"{scene_id}{suffix}"
                else:
                    folder = os.path.join(base_path, self.split, scene_id)
                    fname_lower = f"{prefix}{angle}{suffix}"
                fname_upper = fname_lower.replace(".png", ".PNG")
                path_lower, path_upper = os.path.join(folder, fname_lower), os.path.join(folder, fname_upper)
                f_path = None
                if os.path.exists(path_lower): f_path = path_lower
                elif os.path.exists(path_upper): f_path = path_upper
                if f_path: scene_paths[angle] = f_path
                else: valid_scene = False; break
            if valid_scene: grouped_paths[scene_id] = scene_paths
        return grouped_paths
    def __len__(self): return len(self.valid_scenes)
    def __getitem__(self, idx):
        scene_id = self.valid_scenes[idx]; lr_paths = self.lr_image_paths[scene_id]; hr_paths = self.hr_image_paths[scene_id]; filename = f"scene_{scene_id}_val.png"
        try:
            lr_images_bgr = [cv2.imread(lr_paths[angle], cv2.IMREAD_COLOR) for angle in self.angles]
            hr_images_bgr = [cv2.imread(hr_paths[angle], cv2.IMREAD_COLOR) for angle in self.angles]
            if any(img is None for img in lr_images_bgr) or any(img is None for img in hr_images_bgr): raise IOError("Load fail")
            lr_images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in lr_images_bgr]
            hr_images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in hr_images_bgr]
            lr_np_stacked = np.stack(lr_images_rgb, axis=0); hr_np_stacked = np.stack(hr_images_rgb, axis=0)
            lr_h, lr_w = lr_np_stacked.shape[1:3]; hr_h, hr_w = hr_np_stacked.shape[1:3]
            lr_np_12ch = lr_np_stacked.transpose(0, 3, 1, 2).reshape(12, lr_h, lr_w)
            hr_np_12ch = hr_np_stacked.transpose(0, 3, 1, 2).reshape(12, hr_h, hr_w)
            lr_norm = lr_np_12ch.astype(np.float32) / 255.0; hr_norm = hr_np_12ch.astype(np.float32) / 255.0
            return torch.from_numpy(lr_norm), torch.from_numpy(hr_norm), filename
        except Exception as e: print(f"Error val scene {scene_id}: {e}"); return None, None, filename

# --- Helper Function: RGB to YCbCr ---
def rgb_to_ycbcr(rgb_tensor):
    if rgb_tensor.shape[1] != 3: raise ValueError("Input must be 3 channels (RGB)")
    r, g, b = rgb_tensor[:, 0:1], rgb_tensor[:, 1:2], rgb_tensor[:, 2:3]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cb = -0.1146 * r - 0.3854 * g + 0.5000 * b + 0.5; cr = 0.5000 * r - 0.4542 * g - 0.0458 * b + 0.5
    return torch.clamp(torch.cat([y, cb, cr], dim=1), 0.0, 1.0)

# --- Loss Functions ---
def calculate_stokes(image_12ch):
    if image_12ch.shape[1] != 12: raise ValueError("Input must be 12 channels")
    S0 = (image_12ch[:, 0:3] + image_12ch[:, 6:9] + image_12ch[:, 3:6] + image_12ch[:, 9:12]) / 2
    S1 = image_12ch[:, 0:3] - image_12ch[:, 6:9]
    S2 = image_12ch[:, 3:6] - image_12ch[:, 9:12]
    return S0, S1, S2

class CombinedLoss(nn.Module):
    def __init__(self, intensity_weight=1.0, y_weight=1.0, c_weight=0.1,
                 stokes_weight=1.0):
        super().__init__()
        self.intensity_weight = intensity_weight
        self.y_weight = y_weight
        self.c_weight = c_weight
        self.stokes_weight = stokes_weight

        self.criterion_l1 = nn.L1Loss().to(DEVICE)

    def forward(self, output_12ch, target_12ch):
        # --- 1. Intensity Loss (YCbCr) ---
        loss_intensity_ycbcr = torch.tensor(0.0).to(DEVICE)
        if self.intensity_weight > 0:
            accum_intensity_loss = 0.0
            for i in range(4):
                out_ycbcr = rgb_to_ycbcr(output_12ch[:, i*3:(i+1)*3])
                tgt_ycbcr = rgb_to_ycbcr(target_12ch[:, i*3:(i+1)*3])
                loss_Y  = self.criterion_l1(out_ycbcr[:, 0:1], tgt_ycbcr[:, 0:1])
                loss_CbCr = self.criterion_l1(out_ycbcr[:, 1:3], tgt_ycbcr[:, 1:3])
                accum_intensity_loss += (self.y_weight * loss_Y + self.c_weight * loss_CbCr)
            loss_intensity_ycbcr = accum_intensity_loss / 4.0

        # --- 2. Stokes Loss ---
        loss_stokes = torch.tensor(0.0).to(DEVICE)
        if self.stokes_weight > 0:
            S0_out, S1_out, S2_out = calculate_stokes(output_12ch)
            S0_tgt, S1_tgt, S2_tgt = calculate_stokes(target_12ch)
            loss_stokes = (self.criterion_l1(S0_out, S0_tgt) +
                           self.criterion_l1(S1_out, S1_tgt) +
                           self.criterion_l1(S2_out, S2_tgt)) / 3.0

        total_loss = (self.intensity_weight * loss_intensity_ycbcr +
                      self.stokes_weight * loss_stokes)

        return total_loss, loss_intensity_ycbcr, loss_stokes

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters()); trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad); return total_params, trainable_params
def save_validation_image(tensor_12ch, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    try:
        img_tensor_rgb0 = tensor_12ch.squeeze(0).cpu().detach()[:, :, :].narrow(0, 0, 3)
        img_np_8bit = (torch.clamp(img_tensor_rgb0, 0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, filename), cv2.cvtColor(img_np_8bit, cv2.COLOR_RGB2BGR))
    except Exception as e: print(f"Error saving validation image {filename}: {e}")

# --- Main Training Script ---
if __name__ == "__main__":
    print(f"\nUsing device: {DEVICE}"); print(f"Scale factor: {SCALE_FACTOR}")
    print(f"On-the-fly augmentation: {USE_AUGMENTATION}")
    print(f"Loss Weights: Intensity(YCbCr)={INTENSITY_WEIGHT}, Stokes={STOKES_WEIGHT}")

    # --- Initialize Model ---
    model = model(
        img_size=LR_PATCH_SIZE, 
        in_chans=12, 
        out_chans=12, 
        embed_dim=48,
        depths=[6, 6, 6, 6, 6, 6], 
        num_heads=[6, 6, 6, 6, 6, 6], 
        window_size=8, 
        mlp_ratio=2.
    ).to(DEVICE)
    total_params, trainable_params = count_parameters(model)
    print(f"Model Initialized. Trainable Params: {trainable_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=OPTIMIZER_BETAS, weight_decay=0.0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=COSINE_ETA_MIN)
    criterion_sr = CombinedLoss(intensity_weight=INTENSITY_WEIGHT, y_weight=Y_WEIGHT, c_weight=C_WEIGHT,
                                stokes_weight=STOKES_WEIGHT).to(DEVICE)

    print("Initializing datasets...")
    try:
        train_dataset = DynamicPatchDataset(LR_DATA_PATH, HR_DATA_PATH, 'Train', SCALE_FACTOR, LR_PATCH_SIZE,
                                            NUM_PATCHES_PER_SCENE_TRAIN, HARD_PATCH_MINING_PERCENTAGE,
                                            apply_augmentation=USE_AUGMENTATION)
        val_dataset = PolarizationDatasetVal(LR_DATA_PATH, HR_DATA_PATH, 'Val', SCALE_FACTOR)
    except Exception as e: print(f"Dataset Error: {e}"); sys.exit(1)
    
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Val scenes: {len(val_dataset)}")

    best_val_loss = float('inf'); start_epoch = 0
    train_losses = {'total': [], 'intensity': [], 'stokes': []}
    val_losses_total = []

    if os.path.exists(MODEL_SAVE_PATH):
        try:
            print(f"Resuming training from checkpoint: {MODEL_SAVE_PATH}")
            checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            train_losses = checkpoint.get('train_losses', train_losses)
            val_losses_total = checkpoint.get('val_losses_total', [])
            print(f"Resumed from Epoch {start_epoch}. Best validation loss: {best_val_loss:.5f}")
        except Exception as e: print(f"Could not load checkpoint: {e}. Starting fresh."); start_epoch = 0

    scaler = torch.cuda.amp.GradScaler()
    print("\nStarting Training...")
    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            # --- Generate patches for the current epoch ---
            print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
            train_dataset.generate_epoch_patches()
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
            print(f"Training with {len(train_dataset)} patches...")
            
            model.train()
            running_losses = {'total': 0.0, 'intensity': 0.0, 'stokes': 0.0}
            pbar_train = tqdm(train_loader, desc="[Train]", leave=False)

            for lr_batch, hr_batch in pbar_train:
                lr_batch, hr_batch = lr_batch.to(DEVICE), hr_batch.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast():
                    sr_batch = model(lr_batch)
                    loss_total, loss_int, loss_stk = criterion_sr(sr_batch, hr_batch)
                
                if torch.isnan(loss_total):
                    print("NaN loss detected. Skipping update."); continue
                
                scaler.scale(loss_total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                running_losses['total'] += loss_total.item()
                running_losses['intensity'] += loss_int.item()
                running_losses['stokes'] += loss_stk.item()
                pbar_train.set_postfix({"loss": f"{loss_total.item():.4f}"})
            
            scheduler.step()

            # Log epoch losses
            num_batches = len(train_loader)
            for key in train_losses:
                train_losses[key].append(running_losses[key] / num_batches)

            # Validation loop
            model.eval(); running_val_loss = 0.0
            val_output_dir_epoch = os.path.join(VALIDATION_OUTPUT_PATH, f"epoch_{epoch + 1:03d}")
            pbar_val = tqdm(val_loader, desc="[Val]", leave=False)
            with torch.no_grad():
                for lr_img, hr_img, filename in pbar_val:
                    if lr_img is None: continue
                    lr_img, hr_img = lr_img.to(DEVICE), hr_img.to(DEVICE); filename = filename[0]
                    sr_img_val = model(lr_img)
                    if sr_img_val.shape != hr_img.shape: continue
                    val_loss_sr_comp, _, _ = criterion_sr(sr_img_val, hr_img)
                    running_val_loss += val_loss_sr_comp.item()
                    save_validation_image(sr_img_val, filename, val_output_dir_epoch)
            
            epoch_val_loss = running_val_loss / len(val_loader) if len(val_loader) > 0 else 0
            val_losses_total.append(epoch_val_loss)

            print_msg = (f"Epoch {epoch + 1}/{NUM_EPOCHS} => "
                         f"Train Total: {train_losses['total'][-1]:.5f} | Val Total: {epoch_val_loss:.5f} | "
                         f"LR: {optimizer.param_groups[0]['lr']:.1e}")
            print(print_msg)
            print(f"  Train Breakdown -> Intensity: {train_losses['intensity'][-1]:.5f}, "
                  f"Stokes: {train_losses['stokes'][-1]:.5f}")

            if epoch_val_loss < best_val_loss and len(val_loader) > 0:
                best_val_loss = epoch_val_loss
                os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
                torch.save({
                    'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss, 'train_losses': train_losses, 'val_losses_total': val_losses_total
                }, MODEL_SAVE_PATH)
                print(f"  -> Best model saved (Val Loss: {best_val_loss:.5f})")

    except KeyboardInterrupt: print("\nTraining interrupted.")
    except Exception as e: print(f"\nError during training: {e}"); import traceback; traceback.print_exc()

    print("\nTraining Finished.")

    if start_epoch < NUM_EPOCHS and len(train_losses['total']) > 0:
        epochs_trained_range = range(1, len(train_losses['total']) + 1)
        plt.figure(figsize=(18, 5));
        
        plt.subplot(1, 4, 1);
        plt.plot(epochs_trained_range, train_losses['total'], label='Train Total');
        plt.plot(epochs_trained_range, val_losses_total, label='Val Total');
        plt.title('Total Loss'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

        plt.subplot(1, 4, 2);
        plt.plot(epochs_trained_range, train_losses['intensity'], label='Train Intensity (YCbCr)');
        plt.title('Intensity Loss (Train)'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

        plt.subplot(1, 4, 3);
        plt.plot(epochs_trained_range, train_losses['stokes'], label='Train Stokes');
        plt.title('Stokes Loss (Train)'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

        plt.tight_layout();
        plt.savefig('loss_plot.png');
        print("Loss plots saved.")
