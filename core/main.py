import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import time

# Import project modules
from config import (DEVICE, DATASET_CONFIGS, BATCH_SIZES, NUM_EPOCHS,
                    MODELS_TO_TRAIN, RESULTS_DIR, SEED)
from utils import get_transforms, load_datasets
from model_setup import setup_model
from train_eval import train_model, evaluate_model

def set_seed(seed_value):
    """Sets seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        # Ensure deterministic algorithms are used where possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # Turn off benchmark for determinism

def main():
    set_seed(SEED)
    print(f"Using device: {DEVICE}")
    print(f"Seed set to: {SEED}")

    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Store results for summary and plotting
    all_experiment_results = []
    detailed_run_data = {} # To store loss/acc curves per run

    start_time_total = time.time()

    # --- Experiment Loop ---
    for dataset_type in DATASET_CONFIGS.keys():
        print(f"\n{'='*50}\nProcessing Dataset: {dataset_type.upper()}\n{'='*50}")
        detailed_run_data[dataset_type] = {}

        for batch_size in BATCH_SIZES:
            print(f"\n{'-'*40}\nBatch Size: {batch_size}\n{'-'*40}")
            detailed_run_data[dataset_type][batch_size] = {}

            for model_name in MODELS_TO_TRAIN:
                print(f"\n>>> Training Model: {model_name.upper()} <<<")
                run_start_time = time.time()

                try:
                    # 1. Setup paths and directories for this run
                    model_run_dir = os.path.join(RESULTS_DIR, dataset_type, f"{model_name}_batch{batch_size}")
                    os.makedirs(model_run_dir, exist_ok=True)
                    print(f"Results will be saved in: {model_run_dir}")

                    # 2. Get data transforms
                    base_trans, augment_trans, test_trans = get_transforms(model_name)

                    # 3. Load data
                    train_loader, test_loader, num_classes, class_names = load_datasets(
                        dataset_type, batch_size, base_trans, augment_trans, test_trans
                    )

                    # 4. Setup model
                    model = setup_model(model_name, num_classes)
                    # print(model) # Optional: print model summary

                    # 5. Train model
                    trained_model, train_losses, test_losses, test_accs = train_model(
                        model, train_loader, test_loader, model_name, num_epochs=NUM_EPOCHS
                    )

                    # Store training curves
                    detailed_run_data[dataset_type][batch_size][model_name] = {
                         'train_losses': train_losses,
                         'test_losses': test_losses,
                         'test_accuracies': test_accs
                    }

                    # 6. Evaluate model
                    print(f"\n--- Evaluating {model_name.upper()} on {dataset_type} Test Set ---")
                    eval_metrics = evaluate_model(trained_model, test_loader, class_names, model_run_dir)

                    # 7. Save model weights
                    weights_path = os.path.join(model_run_dir, f'{model_name}_final_weights.pth')
                    torch.save(trained_model.state_dict(), weights_path)
                    print(f"Model weights saved to {weights_path}")

                    # 8. Plot and save training curves for this run
                    plt.figure(figsize=(12, 5))
                    plt.subplot(1, 3, 1)
                    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Train Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title('Training Loss')
                    plt.grid(True)

                    plt.subplot(1, 3, 2)
                    plt.plot(range(1, NUM_EPOCHS + 1), test_losses, label='Test Loss', color='orange')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title('Test Loss')
                    plt.grid(True)

                    plt.subplot(1, 3, 3)
                    plt.plot(range(1, NUM_EPOCHS + 1), test_accs, label='Test Accuracy', color='green')
                    plt.xlabel('Epoch')
                    plt.ylabel('Accuracy (%)')
                    plt.title('Test Accuracy')
                    plt.grid(True)

                    plt.suptitle(f'{model_name.upper()} - {dataset_type} (Batch {batch_size})')
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
                    curves_path = os.path.join(model_run_dir, 'training_curves.png')
                    plt.savefig(curves_path)
                    plt.close()
                    print(f"Training curves saved to {curves_path}")

                    # 9. Store results for final summary
                    best_acc = max(test_accs)
                    best_epoch = test_accs.index(best_acc) + 1
                    all_experiment_results.append({
                        'Dataset': dataset_type,
                        'Batch Size': batch_size,
                        'Model': model_name,
                        'Accuracy': eval_metrics['accuracy'] * 100, # Convert to percentage
                        'Precision (Weighted)': eval_metrics['precision_weighted'],
                        'Recall (Weighted)': eval_metrics['recall_weighted'],
                        'F1 (Weighted)': eval_metrics['f1_weighted'],
                        'Best Accuracy (Epochs)': best_acc,
                        'Best Epoch': best_epoch
                    })

                except Exception as e:
                    print(f"\n****** ERROR occurred for {model_name} on {dataset_type} with batch {batch_size} ******")
                    print(f"Error details: {str(e)}")
                    # Optionally log the error traceback here
                    continue # Skip to the next model/configuration

                run_end_time = time.time()
                print(f">>> Finished {model_name.upper()} in {run_end_time - run_start_time:.2f} seconds <<<")


    # --- Save Summary CSV ---
    summary_file_path = os.path.join(RESULTS_DIR, 'experiment_summary.csv')
    print(f"\nSaving experiment summary to: {summary_file_path}")
    if all_experiment_results:
        # Define the order of columns
        fieldnames = ['Dataset', 'Batch Size', 'Model', 'Accuracy',
                      'Precision (Weighted)', 'Recall (Weighted)', 'F1 (Weighted)',
                      'Best Accuracy (Epochs)', 'Best Epoch']
        try:
            with open(summary_file_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_experiment_results)
            print("Summary CSV saved successfully.")
        except IOError as e:
            print(f"Error writing summary CSV: {e}")
    else:
        print("No results to save in summary CSV.")


    # --- Generate Comparison Plots ---
    print("\nGenerating comparison plots...")
    for dt in DATASET_CONFIGS.keys():
        if dt not in detailed_run_data: continue
        for bs in BATCH_SIZES:
            if bs not in detailed_run_data[dt]: continue

            plt.figure(figsize=(18, 6))

            # Plot Training Loss Comparison
            plt.subplot(1, 3, 1)
            for model_name, data in detailed_run_data[dt][bs].items():
                 if 'train_losses' in data:
                    plt.plot(range(1, len(data['train_losses']) + 1), data['train_losses'], label=model_name)
            plt.title(f'Training Loss - {dt} (Batch {bs})')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend(fontsize='small')
            plt.grid(True)

            # Plot Test Loss Comparison
            plt.subplot(1, 3, 2)
            for model_name, data in detailed_run_data[dt][bs].items():
                if 'test_losses' in data:
                    plt.plot(range(1, len(data['test_losses']) + 1), data['test_losses'], label=model_name)
            plt.title(f'Test Loss - {dt} (Batch {bs})')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend(fontsize='small')
            plt.grid(True)

             # Plot Test Accuracy Comparison
            plt.subplot(1, 3, 3)
            for model_name, data in detailed_run_data[dt][bs].items():
                 if 'test_accuracies' in data:
                    plt.plot(range(1, len(data['test_accuracies']) + 1), data['test_accuracies'], label=model_name)
            plt.title(f'Test Accuracy - {dt} (Batch {bs})')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy (%)')
            plt.legend(fontsize='small')
            plt.grid(True)


            plt.tight_layout()
            comp_plot_path = os.path.join(RESULTS_DIR, f'comparison_{dt}_batch{bs}.png')
            plt.savefig(comp_plot_path)
            plt.close()
            print(f"Comparison plot saved: {comp_plot_path}")

    end_time_total = time.time()
    print(f"\n{'='*50}\nTotal Experiment Time: {(end_time_total - start_time_total) / 60:.2f} minutes\n{'='*50}")

if __name__ == '__main__':
    main()