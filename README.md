# Introducing TH-SE-ResNet for Enhanced Performance and Generalization for Gender Detection using Thermal Images

This repository contains the code for the research paper **Introducing TH-SE-ResNet for Enhanced Performance and Generalization for Gender Detection using Thermal Images** which investigates deep learning models for gender classification using thermal facial images. It includes implementations of baseline CNN architectures (AlexNet, VGG16, ResNet50, InceptionV3, EfficientNet-B0) and a novel proposed architecture (TH-SE-ResNet) evaluated on the Tufts and Charlotte-ThermalFace datasets.

## Features

*   Implementation of various CNN architectures for thermal image classification.
*   Custom `TH-SE-ResNet` model with Channel Input Adapter, Squeeze-and-Excitation blocks and modified fc layer.
*   Data loading and preprocessing pipelines tailored for thermal images (single and multi-channel).
*   Handling of dataset specifics like class imbalance (Tufts) and channel differences.
*   Training script with learning rate warmup and cosine annealing.
*   Evaluation script generating confusion matrices and classification reports.
*   Scripts to run batch experiments across different datasets, models, and batch sizes.

## Requirements

*   Python 3.8+
*   PyTorch 
*   Torchvision 
*   NumPy
*   Matplotlib
*   Scikit-learn
*   Seaborn

Install dependencies using pip:
```bash
pip install -r requirements.txt
```

## Data Setup

1.  **Download Datasets:**
    *   [**Tufts University Thermal Face Dataset**](https://tdface.ece.tufts.edu/)
    *   [**Charlotte-ThermalFace Dataset**](https://github.com/TeCSAR-UNCC/UNCC-ThermalFace)
2.  **Structure Data:** Organize the datasets into the following structure within a base directory (e.g., `./gender_data`):

    ```
    gender_data/
    ├── tufts/
    │   ├── train/
    │   │   ├── female/
    │   │   └── male/
    │   └── test/
    │       ├── female/
    │       └── male/
    ├── charlotte/
    │   ├── train/
    │   │   ├── female/
    │   │   └── male/
    │   └── test/
    │       ├── female/
    │       └── male/
    └── combined/ 
        ├── train/
        │   ├── female/
        │   └── male/
        └── test/
            ├── female/
            └── male/
    ```
*Note: Ensure the `train`/`test` splits are subject-disjoint as described in the paper.*
*Note: The `combined` dataset needs to be prepared beforehand.

3.  **Update `core/config.py`:** Modify the `BASE_DATA_PATH` variable in `core/config.py` if your data directory is located elsewhere.

## Usage

Run the main experiment script:

```bash
python core/main.py
```

This will:
*   Iterate through the datasets, batch sizes, and models defined in `core/config.py`.
*   Train each model configuration.
*   Evaluate the trained model on the corresponding test set.
*   Save results (model weights, training curves, confusion matrix, classification report) into the `./results/` directory, organized by dataset, model, and batch size.
*   Generate comparison plots for loss and accuracy across different models for each dataset/batch size configuration.
*   Save a final `experiment_summary.csv` file in the `./results/` directory.

## Code Structure

*   `core/main.py`: Main script to orchestrate experiments.
*   `core/models.py`: Defines the custom `HybridResNet (TH-SE-ResNet)` and `SEBlock`.
*   `core/model_setup.py`: Handles initialization and configuration of all models (baseline and custom).
*   `core/utils.py`: Contains data loading (`load_datasets`) and image transformation (`get_transforms`) functions.
*   `core/train_eval.py`: Implements the `train_model` and `evaluate_model` functions.
*   `core/config.py`: Stores all configuration variables and hyperparameters.
*   `core/requirements.txt`: Lists package dependencies.
*   `paper.md`: First draft of the research paper.
*   `README.md`: This file.

