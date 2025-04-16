# test.py
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import warnings
from PIL import Image # Needed for BILINEAR/NEAREST constants if used directly
from torchvision import transforms # Needed by dataloader functions
import cv2
import re
from glob import glob
# Import necessary components
from config import Config
from model import *
from loss import *
from metric import calculate_all_metrics
# Import the SINGLE dataset class and transforms from your dataloader.py
from dataloader import UltrasoundSegmentationDataset, JointTransform, Resize, Grayscale, PILToTensor # <- Correct Import
from train import get_model, get_loss_fn, load_checkpoint # Reuse functions from train.py
from utils import plot_metrics_vs_pulses, plot_ablation_area_comparison,postprocess_mask, to_grayscale_numpy

# Suppress specific warnings if needed
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")


def get_test_loader(config):
    """Creates the DataLoader for the test set using the unified dataset."""
    print("--- Creating Test Loader ---")

    # --- Define the same joint transform function used in training ---
    def joint_transform_fn(image, label):
        # Resize first
        image = image.resize(config.IMAGE_SIZE, Image.BILINEAR)
        label = label.resize(config.IMAGE_SIZE, Image.NEAREST)
        # Convert to Grayscale if needed
        if config.IN_CHANNELS == 1:
             image = image.convert('L')
        label = label.convert('L') # Label is always grayscale
        # Convert to Tensor
        image = transforms.ToTensor()(image)
        label = transforms.ToTensor()(label)
        label = (label > 0.5).float()
        return image, label

    # --- Use the single UltrasoundSegmentationDataset ---
    print(f"Using UltrasoundSegmentationDataset (Sequence Length: {config.SEQUENCE_LENGTH})")
    if not hasattr(config, 'TEST_IMAGE_DIR') or not hasattr(config, 'TEST_LABEL_DIR'):
         raise AttributeError("Config needs TEST_IMAGE_DIR and TEST_LABEL_DIR attributes.")

    try:
        dataset = UltrasoundSegmentationDataset(
            image_dir=config.TEST_IMAGE_DIR,
            label_dir=config.TEST_LABEL_DIR,
            transform=joint_transform_fn,
            sequence_length=config.SEQUENCE_LENGTH # Pass sequence length
        )
    except FileNotFoundError as e:
         print(f"ERROR: Data directory not found: {e}")
         raise
    except ValueError as e:
         print(f"ERROR: Problem initializing dataset (e.g., no samples found): {e}")
         raise


    if len(dataset) == 0:
        print(f"WARNING: Test dataset loaded from {config.TEST_IMAGE_DIR} is empty!")

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, # Keep batch_size=1 for sample-wise visualization/saving
        shuffle=False,
        num_workers=getattr(config, 'NUM_WORKERS', 2)
    )
    print(f"Test loader created with {len(dataset)} samples.")
    return test_loader

def extract_2d_slice(tensor):
    """Extracts a 2D NumPy slice from a tensor (handling B, C, H, W)."""
    if tensor is None: return np.zeros((50,50)) # Handle None input
    np_img = tensor.detach().cpu().numpy()
    # Remove Batch and Channel dims if they are 1
    while np_img.ndim > 2 and np_img.shape[0] == 1:
        np_img = np_img[0]
    # If still more than 2D (like C=1 remains), squeeze it
    while np_img.ndim > 2 and np_img.shape[0] == 1: # Squeeze channel if present
         np_img = np_img.squeeze(0)

    if np_img.ndim != 2:
        print(f"Warning: extract_2d_slice received unexpected final shape {np_img.shape}. Check tensor processing.")
        # Attempt a final squeeze just in case
        try:
            np_img = np.squeeze(np_img)
            if np_img.ndim != 2: return np.zeros((50,50)) # Give up if still not 2D
        except:
            return np.zeros((50,50)) # Return dummy if squeeze fails
    return np_img

def evaluate(model, test_loader, criterion, config):
    """Evaluates the model on the test set and saves visualizations."""
    model.eval()
    sample_metrics_list = []
    total_test_loss = 0.0
    num_batches = len(test_loader)
    if num_batches == 0:
        print("ERROR: Test loader has 0 batches. Cannot evaluate.")
        return {}

    vis_folder = os.path.join("test_results", config.EXPERIMENT_NAME, "visualizations")
    os.makedirs(vis_folder, exist_ok=True)
    print(f"Saving visualizations to: {vis_folder}")

    filename_pattern = r't3US(\d+)_(\d+)_(\d+)'

    with torch.no_grad():
        for idx, batch_data in enumerate(tqdm(test_loader, desc="Testing")):
            if not isinstance(batch_data, (list, tuple)) or len(batch_data) != 3:
                print(f"Warning: Skipping malformed test batch {idx+1}/{num_batches}.")
                continue

            data, target, filename = batch_data 
            filename = filename[0]  # batch_size=1
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            expected_dims = 5 if config.SEQUENCE_LENGTH > 1 else 4
            if data.ndim != expected_dims:
                print(f"Warning: Test Batch {idx+1}: Unexpected INPUT data dimension. Got {data.ndim}, expected {expected_dims}. Skipping batch.")
                continue
            if target.ndim != 4:
                print(f"Warning: Test Batch {idx+1}: Unexpected TARGET dimension. Got {target.ndim}, expected 4. Skipping batch.")
                continue

            # Forward pass and compute loss
            pred_logits = model(data)
            loss = criterion(pred_logits, target)
            total_test_loss += loss.item()

            # Extract number of pulses from filename
            match = re.match(filename_pattern, filename)
            pulses = int(match.group(1)) * 20 if match else None

            # Compute metrics and ablation area
            try:
                # pred_prob = torch.sigmoid(pred_logits)
                # pred_binary = (pred_prob > 0.5).float()

                pred_prob = torch.sigmoid(pred_logits)
                pred_binary = (pred_prob > 0.5).float()

                if config.APPLY_POSTPROCESSING:
                    pred_binary = postprocess_mask(pred_binary[0], min_size=config.MIN_COMPONENT_SIZE)
                    pred_binary = pred_binary.unsqueeze(0)  # back to [B, 1, H, W]

                metrics = calculate_all_metrics(pred_logits, target.float(), threshold=0.5)
                metrics['pulses'] = pulses
                metrics['filename'] = filename
                metrics['post_processed'] = config.APPLY_POSTPROCESSING


                # Compute ablation area from prediction
                pred_np = extract_2d_slice(pred_binary)
                pixel_area_mm2 = 0.0025  # Adjust if needed
                ablation_area = np.sum(pred_np) * pixel_area_mm2
                metrics['ablation_area'] = ablation_area

                sample_metrics_list.append(metrics)
            except Exception as e:
                print(f"Error calculating metrics for test batch {idx+1}: {e}")


            # Visualization preparation
            pred_prob = torch.sigmoid(pred_logits)
            pred_binary = (pred_prob > 0.5).float()

            data_vis = data[:, -1] if data.ndim == 5 else data
            img_np = extract_2d_slice(data_vis)
            gt_np = extract_2d_slice(target)
            pred_np = extract_2d_slice(pred_binary)

            # Save visualizations with clear filename
            try:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(to_grayscale_numpy(img_np), cmap='gray', vmin=0, vmax=1); axs[0].set_title("Input Image")
                axs[1].imshow(to_grayscale_numpy(gt_np), cmap='gray', vmin=0, vmax=1); axs[1].set_title("Ground Truth")
                axs[2].imshow(to_grayscale_numpy(pred_np), cmap='gray', vmin=0, vmax=1); axs[2].set_title("Prediction")
                for ax in axs: ax.axis("off")
                plt.suptitle(f"{filename}", fontsize=10)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                base_filename, _ = os.path.splitext(filename)
                plt.savefig(os.path.join(vis_folder, f"{base_filename}_pred.png"), dpi=150, bbox_inches='tight')
            except Exception as e:
                print(f"Error saving visualization for sample {idx+1}: {e}")
            finally:
                plt.close(fig)

    # Save individual metrics clearly for further plotting
    metrics_df = pd.DataFrame(sample_metrics_list)
    metrics_csv_path = os.path.join("test_results", config.EXPERIMENT_NAME, "individual_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Saved individual sample metrics to {metrics_csv_path}")

    # Aggregate average metrics
    if not sample_metrics_list:
        print("ERROR: No metrics calculated.")
        return {"Test_Loss": total_test_loss / num_batches if num_batches > 0 else 0.0}

    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
    avg_metrics = metrics_df[numeric_cols].mean(axis=0, skipna=True).to_dict()
    avg_metrics["Test_Loss"] = total_test_loss / num_batches

    return avg_metrics

# --- save_metrics_to_csv and main remain the same ---

def save_metrics_to_csv(metrics, config):
    """Saves the aggregated test metrics to a CSV file."""
    results_dir = os.path.join("test_results", config.EXPERIMENT_NAME)
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "test_summary_metrics.csv")

    row_data = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Experiment_Name": config.EXPERIMENT_NAME,
        "Model": config.MODEL_NAME,
        "Loss_Function": config.LOSS_FN,
        "Sequence_Length": config.SEQUENCE_LENGTH,
        "Checkpoint": "best.pth.tar", # Assuming best model is used
        **{k: f"{v:.6f}" if isinstance(v, (float, np.number)) and pd.notna(v) else v for k, v in metrics.items()}
    }

    df = pd.DataFrame([row_data])
    file_exists = os.path.isfile(csv_path)
    # Define header dynamically based on keys present in row_data
    header = list(row_data.keys())
    df.to_csv(csv_path, mode='a', header=not file_exists, index=False, columns=header)
    print(f"Saved test metrics summary to {csv_path}")


def main():
    config = Config()

    # --- Define Test Paths in Config --- (Ensure these are set in config.py)
    if not hasattr(config, 'TEST_IMAGE_DIR') or not hasattr(config, 'TEST_LABEL_DIR'):
        print("ERROR: TEST_IMAGE_DIR and TEST_LABEL_DIR must be defined in config.py")
        # Example fallback (NOT RECOMMENDED FOR PRODUCTION):
        # script_dir = os.path.dirname(os.path.abspath(__file__))
        # config.TEST_IMAGE_DIR = os.path.join(script_dir, "../Data/test_Images") # Use NEW folder
        # config.TEST_LABEL_DIR = os.path.join(script_dir, "../Data/test_Labels") # Use NEW folder
        # print(f"Attempting fallback paths:\n  {config.TEST_IMAGE_DIR}\n  {config.TEST_LABEL_DIR}")
        return # Exit if paths are not properly configured


    try:
        test_loader = get_test_loader(config)
        model = get_model(config)
        criterion = get_loss_fn(config)
    except (AttributeError, ValueError, FileNotFoundError, ImportError) as e:
        print(f"Error during setup: {e}")
        print("Please check config.py, data paths, dataloader.py, and model/loss definitions.")
        return

    # --- Load Checkpoint ---
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, config.EXPERIMENT_NAME, "best.pth.tar")
    if os.path.isfile(checkpoint_path):
        load_checkpoint(checkpoint_path, model, None, 0, config.DEVICE) # Pass None for optimizer
    else:
        print(f"ERROR: No checkpoint found at {checkpoint_path}. Cannot run evaluation.")
        return

    # --- Evaluate ---
    print("\n--- Starting Evaluation ---")
    final_metrics = evaluate(model, test_loader, criterion, config)

    # --- Print and Save Results ---
    print("\n--- Average Test Metrics ---")
    if final_metrics:
        # Sort metrics alphabetically for consistent printing
        sorted_metrics = dict(sorted(final_metrics.items()))
        for k, v in sorted_metrics.items():
            print(f"{k}: {v:.4f}" if isinstance(v, (float, np.number)) and pd.notna(v) else f"{k}: {v}")
        save_metrics_to_csv(final_metrics, config)
        #  Assessment of the histotripsy pulse-dependence for the accuracy, Dice Similarity Coefficient, and Hausdorff distance (maximum and mean). 
        metrics_csv_path = os.path.join("test_results", config.EXPERIMENT_NAME, "individual_metrics.csv")
        save_dir = os.path.join("test_results", config.EXPERIMENT_NAME)
        plot_metrics_vs_pulses(metrics_csv_path, save_dir, config.EXPERIMENT_NAME)
        mask_dir = config.TEST_LABEL_DIR
        plot_ablation_area_comparison(
            mask_folder=mask_dir,
            cnn_metrics_path=metrics_csv_path,
            save_path=save_dir,
            experiment_name=config.EXPERIMENT_NAME,
            pixel_area_mm2=0.0025  # Adjust if needed
        )
    else:
        print("Evaluation completed, but no metrics were calculated (check errors above).")

    print("\n--- Testing Finished ---")

if __name__ == "__main__":
    main()