import os
import random
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Imports for both CNN and ResNet models
from .cnn.train import SimpleCNN, prepare_dataloaders as prepare_cnn
from .resnet.train import get_model as get_resnet, prepare_dataloaders as prepare_resnet
from .resnet.train import get_data_transforms, FingerPrintDataset

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Common assets directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(CURRENT_DIR, 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)

# Data directory
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../..', 'fingerprint_data'))

# Number of random probes for identification/authentication
NUM_SAMPLES = 4

# Helper functions

def show_image(ax, img, title=None, success=None):
    arr = img.squeeze().cpu().numpy() if torch.is_tensor(img) else img.convert('L')
    ax.imshow(arr, cmap='gray')
    ax.axis('off')
    if title:
        if success is not None:
            title = f"{title} – {'✔' if success else '✘'}"
        ax.set_title(title)


def evaluate_classification(model, test_loader):
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

    model.eval()
    all_preds, all_labels = [], []
    correct = total = 0
    with torch.no_grad():
        for imgs, labs in test_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            outs = model(imgs)
            _, preds = torch.max(outs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labs.cpu().numpy())
            correct += (preds == labs).sum().item()
            total += labs.size(0)
    accuracy = correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'confusion_matrix': cm}


def perform_identification(model, subset, prefix):
    # subset: a torch.utils.data.Subset of FingerPrintDataset
    orig = subset.dataset
    indices = subset.indices
    fig, axes = plt.subplots(NUM_SAMPLES, 2, figsize=(6, 3*NUM_SAMPLES))
    sample_indices = random.sample(indices, min(NUM_SAMPLES, len(indices)))
    for i, idx in enumerate(sample_indices):
        img, true = orig[idx]
        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(device)).argmax(1).item()
        success = (pred == true)
        # gallery within subset
        gallery_idx = next((j for j in indices if orig.labels[j] == pred), None)
        if gallery_idx is not None:
            gal_img, _ = orig[gallery_idx]
        else:
            gal_img = Image.new('L', img.size)
            success = False
        show_image(axes[i,0], img, title=f"Probe({true})", success=success)
        show_image(axes[i,1], gal_img, title=f"Match({pred})", success=success)
    fig.suptitle(f"1:N Identification ({prefix.upper()})")
    plt.tight_layout(rect=[0,0,1,0.95])
    path = os.path.join(ASSETS_DIR, f"{prefix}_1N.png")
    fig.savefig(path)
    plt.close()


def perform_authentication(model, subset, prefix):
    orig = subset.dataset
    indices = subset.indices
    # Genuine
    fig_g, ax_g = plt.subplots(NUM_SAMPLES, 2, figsize=(6, 3*NUM_SAMPLES))
    sample_indices = random.sample(indices, min(NUM_SAMPLES, len(indices)))
    for i, idx in enumerate(sample_indices):
        img1, lab = orig[idx]
        same = [j for j in indices if orig.labels[j] == lab and j != idx]
        idx2 = same[0] if same else idx
        img2, _ = orig[idx2]
        with torch.no_grad():
            out1 = model(img1.unsqueeze(0).to(device)).argmax(1).item()
            out2 = model(img2.unsqueeze(0).to(device)).argmax(1).item()
        success = (out1 == out2)
        show_image(ax_g[i,0], img1, title='Probe', success=success)
        show_image(ax_g[i,1], img2, title='Genuine', success=success)
    fig_g.suptitle(f"1:1 Genuine ({prefix.upper()})")
    plt.tight_layout(rect=[0,0,1,0.95])
    path_g = os.path.join(ASSETS_DIR, f"{prefix}_Genuine.png")
    fig_g.savefig(path_g)
    plt.close()
    # Impostor
    fig_i, ax_i = plt.subplots(NUM_SAMPLES, 2, figsize=(6, 3*NUM_SAMPLES))
    for i, idx in enumerate(sample_indices):
        img1, lab = orig[idx]
        diff = [j for j in indices if orig.labels[j] != lab]
        idx2 = diff[0] if diff else idx
        img2, _ = orig[idx2]
        with torch.no_grad():
            out1 = model(img1.unsqueeze(0).to(device)).argmax(1).item()
            out2 = model(img2.unsqueeze(0).to(device)).argmax(1).item()
        success = (out1 != out2)
        show_image(ax_i[i,0], img1, title='Probe', success=success)
        show_image(ax_i[i,1], img2, title='Impostor', success=success)
    fig_i.suptitle(f"1:1 Impostor ({prefix.upper()})")
    plt.tight_layout(rect=[0,0,1,0.95])
    path_i = os.path.join(ASSETS_DIR, f"{prefix}_Impostor.png")
    fig_i.savefig(path_i)
    plt.close()


def plot_comparison(results):
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    names = ['Accuracy', 'Precision', 'Recall', 'F1']
    x = range(len(metrics))
    cnn_vals = [results['cnn'][m] for m in metrics]
    res_vals = [results['resnet'][m] for m in metrics]
    plt.figure(figsize=(8, 5))
    plt.bar([i-0.2 for i in x], cnn_vals, width=0.4, label='CNN')
    plt.bar([i+0.2 for i in x], res_vals, width=0.4, label='ResNet')
    plt.xticks(x, names)
    plt.ylim(0, 1)
    plt.legend()
    for i in x:
        plt.text(i-0.2, cnn_vals[i]+0.02, f"{cnn_vals[i]:.2f}", ha='center')
        plt.text(i+0.2, res_vals[i]+0.02, f"{res_vals[i]:.2f}", ha='center')
    plt.title('Model Comparison')
    path = os.path.join(ASSETS_DIR, 'comparison.png')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    results = {}
    # Evaluate CNN
    _, _, test_cnn, c = prepare_cnn(DATA_DIR)
    cnn = SimpleCNN(c).to(device)
    cnn_path = os.path.join(CURRENT_DIR, 'cnn', 'cnn_model.pth')
    cnn.load_state_dict(torch.load(cnn_path, map_location=device))
    res_cnn = evaluate_classification(cnn, test_cnn)
    results['cnn'] = res_cnn
    perform_identification(cnn, test_cnn.dataset, 'cnn')
    perform_authentication(cnn, test_cnn.dataset, 'cnn')
    # Evaluate ResNet
    _, _, test_rs, r = prepare_resnet(DATA_DIR)
    rs = get_resnet(r).to(device)
    rs_path = os.path.join(CURRENT_DIR, 'resnet', 'resnet_model.pth')
    rs.load_state_dict(torch.load(rs_path, map_location=device))
    res_rs = evaluate_classification(rs, test_rs)
    results['resnet'] = res_rs
    perform_identification(rs, test_rs.dataset, 'resnet')
    perform_authentication(rs, test_rs.dataset, 'resnet')
    # Comparison plot
    plot_comparison(results)
    # Print summary
    for name, m in results.items():
        print(f"{name.upper()}: Acc={m['accuracy']:.3f}, Prec={m['precision']:.3f}, Rec={m['recall']:.3f}, F1={m['f1']:.3f}")

if __name__ == '__main__':
    main()
