import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import copy
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from config import DEVICE, LEARNING_RATE, WARMUP_EPOCHS, NUM_EPOCHS

def train_model(model, train_loader, test_loader, model_name, num_epochs=NUM_EPOCHS):
    """Trains the model and returns the best performing version."""
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # Filter parameters that require gradients for the optimizer
    params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params_to_optimize, lr=LEARNING_RATE, betas=(0.9, 0.999))

    # Scheduler setup (Cosine Annealing after Warmup)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - WARMUP_EPOCHS, eta_min=1e-7) # Ensure eta_min is small

    best_acc = 0.0
    best_model_wts = None
    train_losses, test_accuracies = [], []
    test_losses = [] # Track test loss as well

    print(f"Starting training for {model_name}...")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        current_lr = 0.0 # Initialize

        # Learning Rate Warmup
        if epoch < WARMUP_EPOCHS:
            # Linear warmup
            warmup_factor = (epoch + 1) / WARMUP_EPOCHS
            current_lr = LEARNING_RATE * warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        elif epoch == WARMUP_EPOCHS:
             # Ensure LR is exactly LEARNING_RATE at the end of warmup before scheduler takes over
             current_lr = LEARNING_RATE
             for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        else:
             # Scheduler step should happen *after* warmup
             scheduler.step()
             current_lr = optimizer.param_groups[0]['lr'] # Get LR after scheduler step


        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Handle Inception's auxiliary outputs during training
            if model_name == 'inception' and model.training:
                outputs, aux_outputs = model(images)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4 * loss2 # As per Inception paper
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # --- Evaluation Phase ---
        model.eval()
        test_running_loss = 0.0
        correct = 0
        total = 0
        with torch.inference_mode(): # More efficient than torch.no_grad()
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)

                # Calculate test loss (use primary output for Inception)
                if model_name == 'inception' and isinstance(outputs, tuple):
                    test_loss = criterion(outputs[0], labels) # Use main output for eval loss
                else:
                     test_loss = criterion(outputs, labels)
                test_running_loss += test_loss.item()

                # Calculate accuracy
                if model_name == 'inception' and isinstance(outputs, tuple):
                     _, predicted = torch.max(outputs[0].data, 1)
                else:
                    _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_test_loss = test_running_loss / len(test_loader)
        test_losses.append(epoch_test_loss)
        accuracy = 100.0 * correct / total
        test_accuracies.append(accuracy)

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {epoch_loss:.4f} | Test Loss: {epoch_test_loss:.4f} | "
              f"Test Acc: {accuracy:.2f}% | LR: {current_lr:.6f} | "
              f"Time: {epoch_duration:.1f}s")

        # Save the best model based on test accuracy
        if accuracy > best_acc:
            best_acc = accuracy
            # Use state_dict for saving, ensures compatibility
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"  ** New best accuracy: {best_acc:.2f}% **")

    print(f"Training complete. Best Test Accuracy: {best_acc:.2f}%")

    # Load best model weights back into the model
    if best_model_wts:
        model.load_state_dict(best_model_wts)
    else:
        print("Warning: No best model weights saved. Returning last epoch model.")

    return model, train_losses, test_losses, test_accuracies


def evaluate_model(model, test_loader, class_names, save_dir):
    """Evaluates the model and saves confusion matrix and classification report."""
    model.to(DEVICE)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad(): # Use no_grad for evaluation
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)

            # Handle Inception output during eval (only primary)
            if isinstance(outputs, tuple): # Check if Inception output
                outputs = outputs[0]

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy()) # Move labels to CPU as well

    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_str = classification_report(all_labels, all_preds, target_names=class_names)

    # Extract overall accuracy, weighted precision, recall, f1
    accuracy = report_dict['accuracy']
    precision_weighted = report_dict['weighted avg']['precision']
    recall_weighted = report_dict['weighted avg']['recall']
    f1_weighted = report_dict['weighted avg']['f1-score']

    print("\nClassification Report:")
    print(report_str)

    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # Save classification report text file
    report_path = os.path.join(save_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_str)
    print(f"Classification report saved to {report_path}")

    # Return key metrics for summary
    eval_results = {
        'accuracy': accuracy,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'report_str': report_str # Can be useful for detailed logging
    }
    return eval_results