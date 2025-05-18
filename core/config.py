import torch
import os

# --- Data Configuration ---
# Adjust this path to where your 'gender_data' directory is located
BASE_DATA_PATH = "./gender_data"
TUFTS_PATH = os.path.join(BASE_DATA_PATH, 'tufts')
CHARLOTTE_PATH = os.path.join(BASE_DATA_PATH, 'charlotte')
COMBINED_PATH = os.path.join(BASE_DATA_PATH, 'combined')

DATASET_CONFIGS = {
    'tufts': {
        'train_dir': os.path.join(TUFTS_PATH, 'train'),
        'test_dir': os.path.join(TUFTS_PATH, 'test'),
        'augment_minority': True # Special handling for Tufts imbalance
    },
    'charlotte': {
        'train_dir': os.path.join(CHARLOTTE_PATH, 'train'),
        'test_dir': os.path.join(CHARLOTTE_PATH, 'test'),
        'augment_minority': False
    },
    'combined': {
        'train_dir': os.path.join(COMBINED_PATH, 'train'),
        'test_dir': os.path.join(COMBINED_PATH, 'test'),
        'augment_minority': False # Assumes combined is already balanced
    },
    'tufts_to_charlotte': {
        'train_dir': os.path.join(TUFTS_PATH, 'train'),
        'test_dir': os.path.join(CHARLOTTE_PATH, 'test'),
        'augment_minority': True # Augment minority in training set (Tufts)
    },
    'charlotte_to_tufts': {
        'train_dir': os.path.join(CHARLOTTE_PATH, 'train'),
        'test_dir': os.path.join(TUFTS_PATH, 'test'),
        'augment_minority': False # No imbalance in Charlotte training set
    }
}

# --- Model Configuration ---
MODELS_TO_TRAIN = [
    'hybrid_full', 'alexnet', 'vgg', 'resnet', 'inception', 'efficientnet',
    # Ablation study models (optional, uncomment if running)
    # 'hybrid_no_se',
    # 'hybrid_normal_fc',
    # 'hybrid_no_input_conv',
    # 'hybrid_only_fc'
]
NUM_CLASSES = 2 # Binary classification (male/female)

# --- Training Hyperparameters ---
BATCH_SIZES = [32, 64]
LEARNING_RATE = 0.00005  
NUM_EPOCHS = 10
WARMUP_EPOCHS = 5

# --- Hardware ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Output ---
RESULTS_DIR = "./results"

# --- Baseline models Normalization Constants (Used by baseline models) ---
BASELINE_MODEL_MEAN = [0.5, 0.5, 0.5]
BASELINE_MODEL_STD = [0.5, 0.5, 0.5]

# --- ThSE ResNet Normalization Constants ---
THERMAL_MEAN = [0.5]
THERMAL_STD = [0.5]

# --- PyTorch DataLoader Settings ---
NUM_WORKERS = 8
PIN_MEMORY = True
PERSISTENT_WORKERS = True if DEVICE == "cuda" else False # Avoid issues on CPU

# --- Misc ---
SEED = 42 # For reproducibility