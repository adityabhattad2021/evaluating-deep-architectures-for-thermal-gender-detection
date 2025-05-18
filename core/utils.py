import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import transforms, datasets
from config import (DATASET_CONFIGS, BASELINE_MODEL_MEAN, BASELINE_MODEL_STD,
                    THERMAL_MEAN, THERMAL_STD, NUM_WORKERS, PIN_MEMORY,
                    PERSISTENT_WORKERS)

def get_transforms(model_name):
    """Returns appropriate transforms for training and testing based on model type."""

    # Define normalization strategies
    basline_normalize = transforms.Normalize(mean=BASELINE_MODEL_MEAN, std=BASELINE_MODEL_STD)
    thermal_normalize = transforms.Normalize(mean=THERMAL_MEAN, std=THERMAL_STD)

    if model_name in ['alexnet', 'vgg', 'resnet', 'efficientnet']:
        crop_size = 224
        resize_size = 256
        normalize = basline_normalize
        # Basic transforms 
        base_trans = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize
        ])
        # Augmentation transforms for training
        augment_trans = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), # Small blur
            transforms.ToTensor(),
            normalize
        ])
        # Test transforms (usually same as base)
        test_trans = base_trans

    elif model_name == 'inception':
        crop_size = 299
        resize_size = 342 # Standard Inception preprocessing
        normalize = basline_normalize
        base_trans = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize
        ])
        augment_trans = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            normalize
        ])
        test_trans = base_trans

    elif model_name.startswith('hybrid_'):
        crop_size = 224
        resize_size = 256
        normalize = thermal_normalize
        grayscale = True # Hybrid model expects single channel input
        base_trans = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            normalize
        ])
        augment_trans = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size, scale=(0.8, 1.0)),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10,fill=0),
            transforms.RandomAffine(
                degrees=5,  # Rotation range
                translate=(0.05, 0.05),  # light translation
                scale=(0.95, 1.05),  # Slight scaling variations
                fill=0  # Black background for thermal consistency
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            normalize
        ])
        test_trans = base_trans # Keep test simple

    else:
        raise ValueError(f"Transforms not defined for model: {model_name}")

    return base_trans, augment_trans, test_trans

def load_datasets(dataset_type, batch_size, base_transform, augment_transform, test_transform):
    """Loads train and test datasets and creates DataLoaders."""
    if dataset_type not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    config = DATASET_CONFIGS[dataset_type]
    train_dir = config['train_dir']
    test_dir = config['test_dir']
    augment_minority = config.get('augment_minority', False) # Get flag, default False

    # --- Create Training Dataset ---
    # Base dataset (unaugmented or with simple transforms)
    base_dataset = datasets.ImageFolder(train_dir, base_transform)
    class_names = base_dataset.classes
    num_classes = len(class_names)

    datasets_to_combine = [base_dataset]

    # Add augmented version of the entire dataset (standard practice)
    augmented_dataset_full = datasets.ImageFolder(train_dir, augment_transform)
    datasets_to_combine.append(augmented_dataset_full)

    # Specific augmentation for minority class if needed (e.g., Tufts)
    if augment_minority:
        print(f"Applying extra augmentation to minority class for {dataset_type}")
        # Find minority class index (assuming binary, find 'female')
        try:
            female_label_idx = class_names.index('female') # Or determine minority dynamically
            minority_indices = [i for i, (_, label) in enumerate(base_dataset.samples) if label == female_label_idx]

            # Create a dataset with augmentations applied only to minority samples
            minority_augmented_dataset = datasets.ImageFolder(train_dir, augment_transform)
            minority_augmented_subset = Subset(minority_augmented_dataset, minority_indices)
            datasets_to_combine.append(minority_augmented_subset)
            print(f"Added {len(minority_augmented_subset)} extra augmented samples for minority class.")
        except ValueError:
             print("Warning: Could not find 'female' class for minority augmentation.")


    # Combine base and augmented datasets
    train_dataset = ConcatDataset(datasets_to_combine)

    # --- Create Test Dataset ---
    test_dataset = datasets.ImageFolder(test_dir, test_transform)

    # --- Create DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              persistent_workers=PERSISTENT_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                             persistent_workers=PERSISTENT_WORKERS)

    print(f"\nDataset Info for {dataset_type}:")
    print(f"Classes: {class_names}")
    print(f"Train samples (after augmentation combination): {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print("-" * 30)

    return train_loader, test_loader, num_classes, class_names