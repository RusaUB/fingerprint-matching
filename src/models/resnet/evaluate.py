import os
import random
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Local package imports
from .train import get_model, FingerPrintDataset, get_data_transforms

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths and model loading
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = os.path.basename(CURRENT_DIR)
MODEL_PATH = os.path.join(CURRENT_DIR, f"{MODEL_NAME}_model.pth")
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../../fingerprint_data"))

# Load dataset with test transformations
transforms_dict = get_data_transforms()
dataset = FingerPrintDataset(DATA_DIR, transform=transforms_dict['test'])

# Load trained model
model = get_model(dataset.get_num_classes())
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Helper to display images
def show_image(ax, img, title=None):
    if torch.is_tensor(img):
        img_disp = img.squeeze().cpu().numpy()
    else:
        img_disp = img.convert('L')
    ax.imshow(img_disp, cmap='gray')
    ax.axis('off')
    if title:
        ax.set_title(title)

# Number of random probes to evaluate
NUM_SAMPLES = 4
all_indices = list(range(len(dataset)))
probe_indices = random.sample(all_indices, NUM_SAMPLES)

# 1:N Identification
fig_id, axes_id = plt.subplots(NUM_SAMPLES, 2, figsize=(6, 3 * NUM_SAMPLES))
for i, idx in enumerate(probe_indices):
    probe_img, probe_label = dataset[idx]
    with torch.no_grad():
        logits = model(probe_img.unsqueeze(0).to(device))
        pred_label = torch.argmax(logits, dim=1).item()
    gallery_idx = next(j for j, lab in enumerate(dataset.labels) if lab == pred_label)
    gallery_img, _ = dataset[gallery_idx]
    show_image(axes_id[i,0], probe_img, title=f"Probe (True={probe_label})")
    show_image(axes_id[i,1], gallery_img, title=f"Match (Pred={pred_label})")
fig_id.suptitle('1:N Identification')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# 1:1 Genuine Authentication
fig_gen, axes_gen = plt.subplots(NUM_SAMPLES, 2, figsize=(6, 3 * NUM_SAMPLES))
for i, idx in enumerate(probe_indices):
    probe_img, probe_label = dataset[idx]
    same_idxs = [j for j, lab in enumerate(dataset.labels) if lab == probe_label and j != idx]
    genu_idx = random.choice(same_idxs) if same_idxs else idx
    img_genu, _ = dataset[genu_idx]
    with torch.no_grad():
        out1 = model(probe_img.unsqueeze(0).to(device)).argmax(dim=1).item()
        out2 = model(img_genu.unsqueeze(0).to(device)).argmax(dim=1).item()
    match_g = (out1 == out2)
    show_image(axes_gen[i,0], probe_img, title='Probe')
    show_image(axes_gen[i,1], img_genu, title=f'Genuine – Match: {match_g}')
fig_gen.suptitle('1:1 Genuine Authentication')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# 1:1 Impostor Authentication
fig_imp, axes_imp = plt.subplots(NUM_SAMPLES, 2, figsize=(6, 3 * NUM_SAMPLES))
for i, idx in enumerate(probe_indices):
    probe_img, probe_label = dataset[idx]
    other_labels = [lab for lab in dataset.unique_labels if lab != probe_label]
    imp_label = random.choice(other_labels)
    imp_idx = random.choice([j for j, lab in enumerate(dataset.labels) if lab == imp_label])
    img_imp, _ = dataset[imp_idx]
    with torch.no_grad():
        out1 = model(probe_img.unsqueeze(0).to(device)).argmax(dim=1).item()
        out2 = model(img_imp.unsqueeze(0).to(device)).argmax(dim=1).item()
    match_i = (out1 == out2)
    show_image(axes_imp[i,0], probe_img, title='Probe')
    show_image(axes_imp[i,1], img_imp, title=f'Impostor – Match: {match_i}')
fig_imp.suptitle('1:1 Impostor Authentication')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

print("Evaluation complete on multiple samples.")
