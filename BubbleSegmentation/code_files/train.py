# train.py
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard.writer import SummaryWriter  # Import TensorBoard
from tqdm import tqdm
import pandas as pd # Import pandas for easier metrics aggregation
import warnings
import csv # Import csv module
from datetime import datetime # For timestamping logs
from config import Config
from model import * # Imports __init__.py which should import all model classes
from loss import *  # Imports __init__.py which should import all loss classes
# Import the consolidated metrics function
from metric import calculate_all_metrics
from dataloader import create_ultrasound_dataloaders


# Suppress specific warnings if needed
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning)


from model import get_unetpp, get_deeplabv3p, get_fpn, get_torchvision_deeplabv3, load_maskrcnn_model

def get_model(config):
    device = config.DEVICE
    model_name = config.MODEL_NAME
    in_channels = config.IN_CHANNELS
    num_classes = config.NUM_CLASSES
    encoder_name = config.ENCODER_NAME if hasattr(config, 'ENCODER_NAME') else 'resnet34'
    encoder_weights = config.ENCODER_WEIGHTS if hasattr(config, 'ENCODER_WEIGHTS') else 'imagenet'
    use_cuda = config.USE_CUDA if hasattr(config, 'USE_CUDA') else True
    
    print(f"--- Initializing Model: {model_name} ---")
    print(f"Input Channels: {in_channels}, Num Classes: {num_classes}")
    
    if model_name == "ResNet18CNN":
        model = get_resnet18_model(
            in_channels=in_channels,
            num_classes=num_classes,
        )
        print(f"ResNet18CNN pretrained loaded .. ")
        
    elif model_name == "DeepLabV3Plus":
        model = get_deeplabv3p(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes
        )
        print(f"DeepLabV3Plus model initialized with encoder {encoder_name}")
    
    elif model_name == "FPN":
        model = get_fpn(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes
        )
        print(f"FPN model initialized with encoder {encoder_name}")
    
    elif model_name == "TorchvisionDeepLabV3":
        model = get_torchvision_deeplabv3(
            num_classes=num_classes,
            use_cuda=use_cuda
        )
        print(f"DeepLabV3 model initialized from torchvision")
    
    elif model_name == "Unet++":
        model = get_unetpp(
            in_channels=in_channels,
            classes=num_classes
        )
        print(f"Unet++ model initialized")
    
    elif model_name == "MaskRCNN":
        model = load_maskrcnn_model(
            in_channels=in_channels,
            num_classes=num_classes
        )
        print(f"MaskRCNN model loaded")
    
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    
    return model.to(device)

def get_loss_fn(config):
    """Initializes the loss function based on the configuration."""
    loss_fn_name = config.LOSS_FN
    print(f"--- Initializing Loss Function: {loss_fn_name} ---")

    if loss_fn_name == "DiceFocalLoss":
        dice_weight = getattr(config, 'LOSS_DICE_WEIGHT', 0.5)
        focal_weight = getattr(config, 'LOSS_FOCAL_WEIGHT', 0.5)
        criterion = DiceFocalLoss(dice_weight=dice_weight, focal_weight=focal_weight)
        print(f"DiceFocalLoss Params - Dice Weight: {dice_weight}, Focal Weight: {focal_weight}")

    elif loss_fn_name == "DiceLoss":
        criterion = DiceLoss()
        print(f"DiceLoss")

    elif loss_fn_name == "AsymmetricFocalTverskyLoss":
        tversky_weight = getattr(config, 'LOSS_TVERSKY_WEIGHT', 0.5)
        focal_weight = getattr(config, 'LOSS_FOCAL_WEIGHT', 0.5)
        criterion = AsymmetricFocalTverskyLoss(tversky_weight=tversky_weight, focal_weight=focal_weight)
        print(f"AsymmetricFocalTverskyLoss Params - Tversky Weight: {tversky_weight}, Focal Weight: {focal_weight}")
    else:
        raise ValueError(f"Invalid loss function name in config: '{loss_fn_name}'")

    print("--- Loss Function Initialized ---")
    return criterion

def train_one_epoch(model, optimizer, criterion, train_loader, epoch, config, writer):
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} (Train)")
    total_loss = 0.0
    num_batches = len(train_loader)

    for batch_idx, batch_data in enumerate(loop):
        if not isinstance(batch_data, (list, tuple)) or len(batch_data) != 5:
            print(f"Warning: Skipping malformed batch {batch_idx+1}/{num_batches}. Expected 5 items, got: {len(batch_data)}")
            continue

        data, targets, _, _, _ = batch_data
        data, targets = data.to(config.DEVICE), targets.to(config.DEVICE)

        # --- Corruption Checks ---
        if torch.isnan(data).any():
            print(f"[Epoch {epoch+1}][Batch {batch_idx+1}] ⚠ NaNs in input images!")
            continue
        if torch.isnan(targets).any():
            print(f"[Epoch {epoch+1}][Batch {batch_idx+1}] ⚠ NaNs in target labels!")
            continue
        if torch.isinf(data).any():
            print(f"[Epoch {epoch+1}][Batch {batch_idx+1}] ⚠ Infs in input images!")
            continue
        if torch.isinf(targets).any():
            print(f"[Epoch {epoch+1}][Batch {batch_idx+1}] ⚠ Infs in target labels!")
            continue
        if targets.max() > 1 or targets.min() < 0:
            print(f"[Epoch {epoch+1}][Batch {batch_idx+1}] ⚠ Target labels out of range: min={targets.min().item()}, max={targets.max().item()}")
            continue

        # --- Shape Check ---
        expected_dims = 5 if config.MODEL_NAME == "ConvLSTM" else 4
        if data.ndim != expected_dims:
            print(f"[Epoch {epoch+1}][Batch {batch_idx+1}] Unexpected data dimensions: {data.ndim}, expected {expected_dims}")
            continue
        if targets.ndim != (expected_dims - 1):
            print(f"[Epoch {epoch+1}][Batch {batch_idx+1}] Unexpected target dimensions: {targets.ndim}, expected {expected_dims - 1}")
            continue

        optimizer.zero_grad()
        predictions = model(data)
        loss = criterion(predictions, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    writer.add_scalar("Loss/Train", avg_loss, epoch)
    return avg_loss

def validate_one_epoch(model, criterion, val_loader, epoch, config, writer):
    model.eval()
    loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} (Validation)")
    batch_metrics_list = []
    total_val_loss = 0.0
    num_batches = len(val_loader)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loop):
            if not isinstance(batch_data, (list, tuple)) or len(batch_data) != 5:
                print(f"Warning: Skipping malformed validation batch {batch_idx+1}/{num_batches}. Expected 5 items, got: {len(batch_data)}")
                continue

            data, targets, _, _, _ = batch_data
            data, targets = data.to(config.DEVICE), targets.to(config.DEVICE)

            # --- Corruption Checks ---
            if torch.isnan(data).any():
                print(f"[Epoch {epoch+1}][Batch {batch_idx+1}] ⚠ NaNs in input images!")
                continue
            if torch.isnan(targets).any():
                print(f"[Epoch {epoch+1}][Batch {batch_idx+1}] ⚠ NaNs in target labels!")
                continue
            if torch.isinf(data).any():
                print(f"[Epoch {epoch+1}][Batch {batch_idx+1}] ⚠ Infs in input images!")
                continue
            if torch.isinf(targets).any():
                print(f"[Epoch {epoch+1}][Batch {batch_idx+1}] ⚠ Infs in target labels!")
                continue
            if targets.max() > 1 or targets.min() < 0:
                print(f"[Epoch {epoch+1}][Batch {batch_idx+1}] ⚠ Target labels out of range: min={targets.min().item()}, max={targets.max().item()}")
                continue

            # --- Shape Checks ---
            expected_dims = 5 if config.MODEL_NAME == "ConvLSTM" else 4
            if data.ndim != expected_dims:
                print(f"[Epoch {epoch+1}][Batch {batch_idx+1}] Unexpected data dimensions: {data.ndim}, expected {expected_dims}")
                continue
            if targets.ndim != (expected_dims - 1):
                print(f"[Epoch {epoch+1}][Batch {batch_idx+1}] Unexpected target dimensions: {targets.ndim}, expected {expected_dims - 1}")
                continue

            # --- Forward Pass ---
            predictions = model(data)

            if torch.isnan(predictions).any():
                print(f"[Epoch {epoch+1}][Batch {batch_idx+1}] NaNs in predictions!")
            if (predictions.abs() > 1e6).any().item():
                print(f"[Epoch {epoch+1}][Batch {batch_idx+1}] ⚠ Logits too large!")
            if (predictions.abs() < 1e-6).any().item():
                print(f"[Epoch {epoch+1}][Batch {batch_idx+1}] ⚠ Logits too close to zero!")

            loss = criterion(predictions, targets)
            total_val_loss += loss.item()

            try:
                batch_metrics = calculate_all_metrics(predictions, targets, threshold=0.5)
                batch_metrics_list.append(batch_metrics)
            except Exception as e:
                print(f"Error calculating metrics for validation batch {batch_idx+1}: {e}")

            loop.set_postfix(loss=loss.item())

    if not batch_metrics_list:
        print("Warning: No metrics calculated during validation.")
        return (total_val_loss / num_batches if num_batches > 0 else 0.0), {}

    metrics_df = pd.DataFrame(batch_metrics_list)
    avg_metrics_dict = metrics_df.mean(axis=0).to_dict()
    avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0.0

    writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
    for key, value in avg_metrics_dict.items():
        tag_name = key.replace(" ", "_").replace("(", "").replace(")", "")
        if pd.notna(value) and isinstance(value, (float, int, np.number)):
            writer.add_scalar(f"Metrics/{tag_name}", value, epoch)
        else:
            print(f"Warning: Could not log metric '{key}' with value '{value}' (type: {type(value)})")

    return avg_val_loss, avg_metrics_dict


def save_checkpoint(model, optimizer, filename):
    """Saves checkpoint."""
    try:
        print(f"=> Saving checkpoint to {filename}")
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(checkpoint, filename)
    except Exception as e:
        print(f"Error saving checkpoint to {filename}: {e}")

def load_checkpoint(checkpoint_file, model, optimizer, lr, device):
    """Loads checkpoint."""
    if not os.path.isfile(checkpoint_file):
        print(f"=> Checkpoint file not found at {checkpoint_file}. Skipping load.")
        return
    print(f"=> Loading checkpoint from {checkpoint_file}")
    try:
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        if optimizer is not None and "optimizer" in checkpoint:
             optimizer.load_state_dict(checkpoint["optimizer"])
             for param_group in optimizer.param_groups:
                 param_group["lr"] = lr # Reset LR from config
        print("=> Checkpoint loaded successfully")
    except Exception as e:
        print(f"=> Error loading checkpoint: {e}")


# --- NEW: CSV Logging Function ---
def log_metrics_to_csv(log_path, epoch, config, train_loss, val_loss, metrics_dict):
    """Appends metrics and config details for an epoch to a CSV file."""
    file_exists = os.path.isfile(log_path)
    # Define header including essential config and all metric keys
    header = [
        'Timestamp', 'Epoch', 'Experiment_Name', 'Model_Name', 'Optimizer', 
        'Loss_Function','Learning_Rate', 'Batch_Size',
        'Train_Loss', 'Validation_Loss'
    ]
    # Dynamically add metric keys from the dictionary, ensuring order
    metric_keys = sorted([key for key in metrics_dict.keys() if pd.notna(metrics_dict[key])]) # Filter out potential NaNs
    header.extend(metric_keys)

    # Prepare data row, converting values to strings for CSV
    data_row = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Epoch': epoch + 1, # Use 1-based epoch for logging
        'Experiment_Name': config.EXPERIMENT_NAME,
        'Model_Name': config.MODEL_NAME,
        'Optimizer': config.OPTIMIZER,
        'Loss_Function': config.LOSS_FN,
        'Learning_Rate': f"{config.LEARNING_RATE:.1E}", # Scientific notation
        'Batch_Size': config.BATCH_SIZE,
        'Train_Loss': f"{train_loss:.6f}",
        'Validation_Loss': f"{val_loss:.6f}"
    }
    # Add formatted metrics only for the valid keys
    for key in metric_keys:
         value = metrics_dict[key]
         data_row[key] = f"{value:.6f}" if isinstance(value, (float, np.number)) else str(value)

    # Write to CSV
    try:
        with open(log_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header, extrasaction='ignore') # Ignore extra keys not in header
            if not file_exists or os.path.getsize(log_path) == 0: # Check size too
                writer.writeheader() # Write header only if file is new or empty
            writer.writerow(data_row)
    except IOError as e:
        print(f"Error writing to CSV log file {log_path}: {e}")
    except Exception as e:
         print(f"An unexpected error occurred during CSV logging: {e}")


def main():
    config = Config() # Load configuration

    train_loader, val_loader = create_ultrasound_dataloaders(
    # image_dir=config.IMAGE_DIR,
    # label_dir=config.LABEL_DIR,e
    # batch_size=config.BATCH_SIZE,
    # image_size=config.IMAGE_SIZE,
    # sequence_length=config.SEQUENCE_LENGTH,
    # use_augmentation=config.USE_AUGMENTATION
)

    # length of dataset - train - test - val
        


    # --- Setup Directories ---
    experiment_dir = os.path.join(config.LOG_DIR, config.EXPERIMENT_NAME)
    model_ckpt_dir = os.path.join(config.CHECKPOINT_DIR, config.EXPERIMENT_NAME)
    os.makedirs(model_ckpt_dir, exist_ok=True)
    os.makedirs(experiment_dir, exist_ok=True)

    # --- CSV Log File Path ---
    csv_log_path = os.path.join(experiment_dir, config.CSV_LOG_FILE)
    print(f"CSV metrics log will be saved to: {csv_log_path}")

    

    if not hasattr(config, 'VISUALIZE_EVERY'):
        config.VISUALIZE_EVERY = 5
    if not hasattr(config, 'SAVE_MODEL'): # Add default if missing
        config.SAVE_MODEL = True


    # --- Initialize Model, Loss, Optimizer ---
    model = get_model(config)
    criterion = get_loss_fn(config)
    # optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    if config.OPTIMIZER == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.SGD_MOMENTUM
        )
        if config.USE_SCHEDULER:
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.SCHEDULER_STEP_SIZE,
                gamma=config.SCHEDULER_GAMMA
            )
        else:
            scheduler = None
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE
        )
        scheduler = None

    # --- TensorBoard Writer ---
    writer = SummaryWriter(log_dir=experiment_dir)
    print(f"TensorBoard logs will be saved in: {experiment_dir}")
    print(f"Checkpoints will be saved in: {model_ckpt_dir}")


    # --- Training Loop ---
    best_val_loss = float('inf')
    epochs_no_improve = 0 # Counter for early stopping

    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, optimizer, criterion, train_loader, epoch, config, writer)
        val_loss, avg_val_metrics = validate_one_epoch(model, criterion, val_loader, epoch, config, writer)

        if scheduler is not None:
            scheduler.step()


        print(f"\n--- Epoch {epoch+1}/{config.NUM_EPOCHS} ---")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Check if metrics dictionary is not empty before printing/logging
        if avg_val_metrics:
            print("Average Validation Metrics:")
            for key, value in avg_val_metrics.items():
                # Handle potential non-numeric values gracefully during print
                try: print(f"  {key}: {float(value):.4f}")
                except (ValueError, TypeError): print(f"  {key}: {value}")

            # --- Log Metrics to CSV --- (Moved inside the check)
            log_metrics_to_csv(csv_log_path, epoch, config, train_loss, val_loss, avg_val_metrics)
        else:
             print("Validation metrics could not be calculated for this epoch.")


        # --- Save Checkpoints ---
        if config.SAVE_MODEL:
            # Save checkpoint periodically (e.g., every 2 epochs)
            if (epoch + 1) % 2 == 0 :
                ckpt_path = os.path.join(model_ckpt_dir, f"epoch_{epoch+1}.pth.tar")
                save_checkpoint(model, optimizer, filename=ckpt_path)

        # --- Visualize Predictions ---
        if (epoch + 1) % config.VISUALIZE_EVERY == 0:
            visualize_predictions(model, val_loader, config, epoch, writer, num_samples=10)

    best_model_path = 'best_model.pt'
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path} for final evaluation...")
        model.load_state_dict(torch.load(best_model_path))

    writer.close()
    print("--- Training Finished ---")
    print(f"CSV log saved: {csv_log_path}")


def visualize_predictions(model, val_loader, config, epoch, writer, num_samples=10):
    """Visualizes predictions and saves/logs them."""
    print(f"--- Visualizing Predictions for Epoch {epoch+1} ---")
    model.eval()
    images_shown = 0
    # Ensure num_samples is positive
    num_samples = max(1, num_samples) # Show at least 1 if possible

    # Create figure with specified layout: num_samples rows, 3 columns
    fig, axs = plt.subplots(num_samples, 3, figsize=(9, num_samples * 3)) # Width=9 (3*3), Height=num_samples*3
    # Ensure axs is always a 2D array, even if num_samples is 1
    if num_samples == 1:
        axs = np.array([axs]) # Wrap in another array dimension

    # Turn off all axes initially
    for ax_row in axs:
         for ax in ax_row:
             ax.axis("off")

    with torch.no_grad():
        # Iterate through val_loader until enough samples are collected or loader ends
        for batch_idx, batch_data in enumerate(val_loader):
            if images_shown >= num_samples: break

            # Basic batch integrity check
            if not isinstance(batch_data, (list, tuple)) or len(batch_data) != 3: continue
            data, targets ,filename = batch_data
            data, targets = data.to(config.DEVICE), targets.to(config.DEVICE)

             # Basic shape check (only for non-sequential for now)
            if config.SEQUENCE_LENGTH <= 1 and (data.ndim != 4 or targets.ndim != 4): continue

            # --- Input Shape Handling --- (Keep for future ConvLSTM use)
            is_sequential = data.ndim == 5
            if is_sequential:
                data_vis = data[:, -1, :, :, :] # Visualize last frame
            else:
                data_vis = data

            # --- Get Model Output ---
            outputs_raw = model(data)
            outputs_prob = torch.sigmoid(outputs_raw)
            outputs_binary = (outputs_prob > 0.5).int()

            # --- Plot samples from the current batch ---
            for i in range(data_vis.shape[0]):
                if images_shown >= num_samples: break

                # img_np = data_vis[i].cpu().numpy()
                # label_np = targets[i].cpu().numpy()
                # pred_np_binary = outputs_binary[i].cpu().numpy()


                img_np = extract_2d_slice(data_vis[i])
                label_np = extract_2d_slice(targets[i])
                pred_np_binary = extract_2d_slice(outputs_binary[i])

                # Plot Image in column 0 of the current row
                axs[images_shown, 0].imshow(to_grayscale_numpy(img_np), cmap='gray', vmin=0, vmax=1)
                axs[images_shown, 0].set_title(f"Image {images_shown+1}")
                axs[images_shown, 0].axis("off") # Re-enable axis for this plot

                # Plot Ground Truth in column 1
                axs[images_shown, 1].imshow(to_grayscale_numpy(label_np), cmap='gray', vmin=0, vmax=1)
                axs[images_shown, 1].set_title(f"GT {images_shown+1}")
                axs[images_shown, 1].axis("off")

                # Plot Prediction in column 2
                axs[images_shown, 2].imshow(to_grayscale_numpy(pred_np_binary), cmap='gray', vmin=0, vmax=1)
                axs[images_shown, 2].set_title(f"Pred {images_shown+1}")
                axs[images_shown, 2].axis("off")

                images_shown += 1

    # Adjust layout if fewer images were shown than planned
    if images_shown < num_samples:
        print(f"Warning: Only able to visualize {images_shown} samples (requested {num_samples}).")
        # Optional: Adjust figure size or remove empty rows if desired
        # For simplicity, we just leave empty rows blank as axes are off

    plt.tight_layout(pad=0.5)

    # --- Save figure and log to TensorBoard ---
    vis_log_dir = os.path.join(config.LOG_DIR, config.EXPERIMENT_NAME, "visualizations")
    os.makedirs(vis_log_dir, exist_ok=True)
    save_path = os.path.join(vis_log_dir, f"epoch_{epoch+1}_predictions.png")

    try:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved validation visualization to {save_path}")
        if writer:
            # Read image and log to TensorBoard
            try:
                img = plt.imread(save_path)
                if img.ndim == 3: img = np.transpose(img[:, :, :3], (2, 0, 1)) # HWC -> CHW (RGB)
                elif img.ndim == 2: img = np.expand_dims(img, axis=0) # HW -> CHW (Grayscale)
                writer.add_image(f"Validation_Predictions/Epoch_{epoch+1}", img, global_step=epoch)
            except FileNotFoundError:
                 print(f"Error: Could not read saved image {save_path} for TensorBoard logging.")
    except Exception as e:
        print(f"Error saving or logging visualization: {e}")
    finally:
        plt.close(fig) # Close the figure


def extract_2d_slice(tensor):
    np_img = tensor.cpu().numpy()
    while np_img.ndim > 2:
        np_img = np_img[0]
    return np_img


if __name__ == "__main__":
    main()