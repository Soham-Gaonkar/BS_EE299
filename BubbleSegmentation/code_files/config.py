# config.py
import os
import torch

class Config:
    # General
    IN_CHANNELS = 1
    IMAGE_SIZE = (1024, 256)
    NUM_CLASSES = 1

    IMAGE_DIR = "../Data/US_2"
    LABEL_DIR = "../Data/Labels_2"
    TEST_IMAGE_DIR = "../Data/US_Test_2023April7" 
    TEST_LABEL_DIR = "../Data/Labels_Test_2023April7" 

    # Training
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", 3e-4)) # 1e-4 
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))             
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 6))           

    # Optimizer
    OPTIMIZER = os.getenv("OPTIMIZER", "Adam")
    USE_SCHEDULER = os.getenv("USE_SCHEDULER", "True").lower() == "true"
    SGD_MOMENTUM = float(os.getenv("SGD_MOMENTUM", 0.9))
    SCHEDULER_STEP_SIZE = int(os.getenv("SCHEDULER_STEP_SIZE", 20))
    SCHEDULER_GAMMA = float(os.getenv("SCHEDULER_GAMMA", 0.5))# 0.1

    # Model
    # Options: ResNet18CNN, AttentionUNet, DeepLabV3Plus, ConvLSTM , SimpleUNetMini
    MODEL_NAME = os.getenv("MODEL_NAME", "DeepLabV3Plus")
   
    # Loss
    # Options: DiceFocalLoss, DiceLoss, AsymmetricFocalTverskyLoss, SoftIoULoss
    LOSS_FN = os.getenv("LOSS_FN", "AsymmetricFocalTverskyLoss")
    LOSS_DICE_WEIGHT = os.getenv("LOSS_DICE_WEIGHT", 0.5)
    LOSS_TVERSKY_WEIGHT = os.getenv("LOSS_TVERSKY_WEIGHT", 0.5)
    LOSS_FOCAL_WEIGHT = os.getenv("LOSS_FOCAL_WEIGHT", 0.5)

    # Early stopping settings
    EARLY_STOPPING = True
    PATIENCE = 5            # Number of epochs to wait before stopping
    DELTA = 1e-4           # Minimum change in validation loss to qualify as improvement

    # Logging
    SAVE_MODEL = os.getenv("SAVE_MODEL", "True").lower() == "true"
    CHECKPOINT_DIR = "checkpoints/"
    LOG_DIR = "logs/"
    EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", f"{MODEL_NAME}_{LOSS_FN}_Epochs{NUM_EPOCHS}_LR{LEARNING_RATE}")
    VISUALIZE_EVERY = int(os.getenv("VISUALIZE_EVERY", 4))
    CSV_LOG_FILE = "training_log.csv"