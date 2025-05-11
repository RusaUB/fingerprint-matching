import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_metrics_to_excel(train_losses, val_losses, train_accs, val_accs, test_metrics, filename="fingerprint_metrics.xlsx"):
    """
    Saves training, validation, and test metrics to an Excel file.

    Args:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        train_accs (list): List of training accuracies per epoch.
        val_accs (list): List of validation accuracies per epoch.
        test_metrics (dict): Dictionary containing test metrics ('precision', 'recall', 'f1', 'confusion_matrix').
        filename (str): Name of the Excel file to save.
    """
    try:
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            # Sheet 1: Epoch-wise Training and Validation Metrics
            epochs = list(range(1, len(train_losses) + 1))
            epoch_metrics_df = pd.DataFrame({
                'Epoch': epochs,
                'Training Loss': train_losses,
                'Validation Loss': val_losses,
                'Training Accuracy': train_accs,
                'Validation Accuracy': val_accs
            })
            epoch_metrics_df.to_excel(writer, sheet_name='Epoch Metrics', index=False)

            # Sheet 2: Test Set Summary Metrics
            summary_metrics_df = pd.DataFrame({
                'Metric': ['Precision (weighted)', 'Recall (weighted)', 'F1-Score (weighted)'],
                'Value': [test_metrics['precision'], test_metrics['recall'], test_metrics['f1']]
            })
            summary_metrics_df.to_excel(writer, sheet_name='Test Summary', index=False)

            # Sheet 3: Confusion Matrix
            cm = test_metrics['confusion_matrix']
            # Use generic labels for CM rows/columns (e.g., True_0, Pred_0)
            # If you have access to idx_to_label map, you could use actual class names.
            num_classes_cm = cm.shape[0]
            cm_index = [f'True Class {i}' for i in range(num_classes_cm)]
            cm_columns = [f'Predicted Class {i}' for i in range(num_classes_cm)]
            cm_df = pd.DataFrame(cm, index=cm_index, columns=cm_columns)
            cm_df.to_excel(writer, sheet_name='Confusion Matrix') # index=True and columns=True by default here

        print(f"Metrics successfully saved to {filename}")

    except Exception as e:
        print(f"Error saving metrics to Excel: {e}")
        print("Please ensure you have 'pandas' and 'xlsxwriter' installed ('pip install pandas xlsxwriter').")


def plot_metrics(train_losses, val_losses, train_accs, val_accs, test_metrics, model_name, save_dir):
    
    """
    Plot and save each metric separately as individual figures:
    1. Training Loss vs Validation Loss
    2. Training Accuracy vs Validation Accuracy
    3. Bar chart of Precision, Recall, F1-Score
    4. Confusion Matrix heatmap (first 10 classes if large)
    """
    # 1. Training and Validation Loss
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    loss_path = os.path.join(save_dir, f'{model_name}_loss_curve.png')
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.show()
    print(f"Saved plot: {loss_path}")

    # 2. Training and Validation Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    acc_path = os.path.join(save_dir, f'{model_name}_accuracy_curve.png')
    plt.tight_layout()
    plt.savefig(acc_path)
    plt.show()
    print(f"Saved plot: {acc_path}")

    # 3. Precision, Recall, F1-Score Bar Chart
    plt.figure(figsize=(6, 6))
    metrics = ['Precision', 'Recall', 'F1-Score']
    values = [test_metrics['precision'], test_metrics['recall'], test_metrics['f1']]
    plt.bar(metrics, values)
    plt.ylim(0, 1)
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    plt.ylabel('Score')
    plt.title('Test Set Metrics')
    prf_path = os.path.join(save_dir, f'{model_name}_precision_recall_f1.png')
    plt.tight_layout()
    plt.savefig(prf_path)
    plt.show()
    print(f"Saved plot: {prf_path}")

    # 4. Confusion Matrix Heatmap
    cm = test_metrics['confusion_matrix']
    if cm.shape[0] > 10:
        cm_display = cm[:10, :10]
        title = 'Confusion Matrix (First 10 Classes)'
        cm_path = os.path.join(save_dir, f'{model_name}confusion_matrix_top10.png')
    else:
        cm_display = cm
        title = 'Confusion Matrix'
        cm_path = os.path.join(save_dir, f'{model_name}_confusion_matrix.png')

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_display, annot=False, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.show()
    print(f"Saved plot: {cm_path}")
