import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image, ImageOps
import random
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import itertools

from ...utils import save_metrics_to_excel, plot_metrics


# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
IMAGE_SIZE = (288, 384)
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
EMBEDDING_DIM = 128  # Dimension of the embedding vector
MARGIN = 1.0  # Margin for contrastive loss

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = os.path.basename(CURRENT_DIR)
ASSETS_DIR = os.path.join(CURRENT_DIR, "assets")

if not os.path.exists(ASSETS_DIR):
    os.makedirs(ASSETS_DIR)

class FingerPrintDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []  # Person IDs
        
        # Get all TIFF files
        for filename in os.listdir(data_dir):
            if filename.endswith('.tif'):
                # Parse filename to extract person ID (xxx)
                match = re.match(r'(\d+)_\d+_\d+\.tif', filename)
                if match:
                    person_id = int(match.group(1))
                    self.image_paths.append(os.path.join(data_dir, filename))
                    self.labels.append(person_id)
        
        # Get unique person IDs and map them to indices
        self.unique_labels = sorted(list(set(self.labels)))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Convert all labels to indices
        self.labels = [self.label_to_idx[label] for label in self.labels]
        
        print(f"Found {len(self.image_paths)} images from {len(self.unique_labels)} different people")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image and convert to grayscale
        image = Image.open(image_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_num_classes(self):
        return len(self.unique_labels)


class SiameseDataset(Dataset):
    def __init__(self, dataset, pairs_per_class=10):
        self.dataset = dataset
        self.pairs_per_class = pairs_per_class
        self.pairs = self._generate_pairs()
        
    def _generate_pairs(self):
        pairs = []
        labels_to_indices = {}
        
        # Group indices by labels
        for idx, (_, label) in enumerate(self.dataset):
            if label not in labels_to_indices:
                labels_to_indices[label] = []
            labels_to_indices[label].append(idx)
        
        # Generate positive pairs (same class)
        for label, indices in labels_to_indices.items():
            if len(indices) < 2:
                continue
                
            # Generate positive pairs (limit to pairs_per_class per class)
            pairs_count = min(self.pairs_per_class, len(indices) * (len(indices) - 1) // 2)
            positive_pairs = random.sample(list(itertools.combinations(indices, 2)), pairs_count)
            for idx1, idx2 in positive_pairs:
                pairs.append((idx1, idx2, 1))  # 1 means same class
        
        # Generate negative pairs (different classes)
        negative_pairs_count = len(pairs)  # Balance the dataset with equal negative pairs
        class_pairs = list(itertools.combinations(labels_to_indices.keys(), 2))
        
        if not class_pairs:
            return pairs
            
        negative_counter = 0
        while negative_counter < negative_pairs_count:
            # Select two different classes
            c1, c2 = random.choice(class_pairs)
            
            # Select one random index from each class
            idx1 = random.choice(labels_to_indices[c1])
            idx2 = random.choice(labels_to_indices[c2])
            
            pairs.append((idx1, idx2, 0))  # 0 means different class
            negative_counter += 1
            
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        idx1, idx2, label = self.pairs[idx]
        img1, _ = self.dataset[idx1]
        img2, _ = self.dataset[idx2]
        
        return img1, img2, torch.tensor(label, dtype=torch.float)


class CustomRotationTransform:
    def __init__(self, max_angle=15):
        self.max_angle = max_angle
        
    def __call__(self, img):
        if random.random() < 0.5:
            angle = random.uniform(-self.max_angle, self.max_angle)
            return img.rotate(angle, resample=Image.BILINEAR, expand=False)
        return img

class CustomShiftTransform:
    def __init__(self, max_shift=5):
        self.max_shift = max_shift
        
    def __call__(self, img):
        if random.random() < 0.5:
            shift_x = random.randint(-self.max_shift, self.max_shift)
            shift_y = random.randint(-self.max_shift, self.max_shift)
            return ImageOps.expand(img, border=(shift_x, shift_y, 0, 0), fill=0)
        return img

class NoiseTransform:
    def __init__(self, noise_factor=0.05):
        self.noise_factor = noise_factor
        
    def __call__(self, img):
        if random.random() < 0.3:
            img_array = np.array(img)
            noise = np.random.normal(0, self.noise_factor * 255, img_array.shape)
            noisy_img = img_array + noise
            noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
            return Image.fromarray(noisy_img)
        return img

class ContrastTransform:
    def __init__(self, factor_range=(0.8, 1.2)):
        self.factor_range = factor_range
        
    def __call__(self, img):
        if random.random() < 0.3:
            factor = random.uniform(*self.factor_range)
            return ImageOps.autocontrast(img, cutoff=factor)
        return img

def get_data_transforms():
    # Define data transformations
    data_transforms = {
        'train': transforms.Compose([
            CustomRotationTransform(15),
            CustomShiftTransform(5),
            NoiseTransform(),
            ContrastTransform(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    }
    return data_transforms

def prepare_dataloaders(data_dir):
    data_transforms = get_data_transforms()
    
    # Create dataset
    full_dataset = FingerPrintDataset(data_dir, transform=data_transforms['train'])
    
    # Split dataset
    dataset_size = len(full_dataset)
    test_size = int(TEST_SPLIT * dataset_size)
    val_size = int(VALIDATION_SPLIT * dataset_size)
    train_size = dataset_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Override transformations for validation and test sets
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']
    test_dataset.dataset.transform = data_transforms['test']
    
    # Create Siamese datasets
    siamese_train_dataset = SiameseDataset(train_dataset, pairs_per_class=20)
    siamese_val_dataset = SiameseDataset(val_dataset, pairs_per_class=10)
    siamese_test_dataset = SiameseDataset(test_dataset, pairs_per_class=10)
    
    # Create data loaders
    train_loader = DataLoader(siamese_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(siamese_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(siamese_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Also create a loader for the raw test dataset (for identification evaluation)
    raw_test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, raw_test_loader, full_dataset.get_num_classes()

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        # Use a smaller CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, EMBEDDING_DIM)
        )
        
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # L2 normalize the embeddings
        x = F.normalize(x, p=2, dim=1)
        return x

class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.embedding_net = EmbeddingNet()
        
    def forward_one(self, x):
        return self.embedding_net(x)
    
    def forward(self, x1, x2):
        output1 = self.forward_one(x1)
        output2 = self.forward_one(x2)
        return output1, output2
    
    def get_embedding(self, x):
        return self.forward_one(x)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=MARGIN):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):
        # Calculate Euclidean distance
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        # Contrastive loss
        loss_contrastive = torch.mean(
            label * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        
        return loss_contrastive

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for img1, img2, labels in train_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            # Forward pass
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * img1.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                
                output1, output2 = model(img1, img2)
                loss = criterion(output1, output2, labels)
                
                running_loss += loss.item() * img1.size(0)
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {epoch_train_loss:.4f}, '
              f'Val Loss: {epoch_val_loss:.4f}')
        
        # Save the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(CURRENT_DIR, f"{MODEL_NAME}_model.pth"))
    
    # Load the best model
    model.load_state_dict(torch.load(os.path.join(CURRENT_DIR, f"{MODEL_NAME}_model.pth")))
    
    return model, train_losses, val_losses

def evaluate_siamese_model(model, test_loader, threshold=0.5):
    model.eval()
    correct = 0
    total = 0
    all_distances = []
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for img1, img2, labels in test_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            output1, output2 = model(img1, img2)
            euclidean_distance = F.pairwise_distance(output1, output2)
            
            # If distance < threshold, predict same class (1), else different class (0)
            predictions = (euclidean_distance < threshold).float()
            
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            
            all_distances.extend(euclidean_distance.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())
    
    accuracy = correct / total
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'distances': all_distances,
        'confusion_matrix': cm
    }

def find_optimal_threshold(model, val_loader):
    model.eval()
    all_distances = []
    all_labels = []
    
    with torch.no_grad():
        for img1, img2, labels in val_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            output1, output2 = model(img1, img2)
            euclidean_distance = F.pairwise_distance(output1, output2)
            
            all_distances.extend(euclidean_distance.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Try different thresholds and find the one with the best F1 score
    thresholds = np.arange(0.1, 2.0, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        predictions = (np.array(all_distances) < threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            all_labels, predictions, average='binary', zero_division=0
        )
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"Optimal threshold: {best_threshold} with F1: {best_f1:.4f}")
    return best_threshold

def identification_accuracy(model, test_loader, gallery_loader=None):
    """
    Evaluate the model's ability to identify the correct person
    by comparing a query fingerprint against a gallery of known fingerprints
    """
    model.eval()
    
    if gallery_loader is None:
        gallery_loader = test_loader
    
    # First, create a gallery of embeddings
    gallery_embeddings = []
    gallery_labels = []
    
    with torch.no_grad():
        for images, labels in gallery_loader:
            images = images.to(device)
            embeddings = model.get_embedding(images)
            gallery_embeddings.append(embeddings)
            gallery_labels.append(labels)
    
    gallery_embeddings = torch.cat(gallery_embeddings, dim=0).cpu()
    gallery_labels = torch.cat(gallery_labels, dim=0).cpu()
    
    # Now evaluate on the test set
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            embeddings = model.get_embedding(images)
            
            # For each query embedding, find the closest gallery embedding
            for i, embedding in enumerate(embeddings):
                distances = torch.norm(gallery_embeddings - embedding.cpu(), dim=1)
                min_idx = torch.argmin(distances)
                predicted_label = gallery_labels[min_idx]
                
                if predicted_label == labels[i].cpu():
                    correct += 1
                total += 1
    
    return correct / total

def plot_siamese_metrics(train_losses, val_losses, test_metrics, model_name, save_dir):
    """
    Plot training and validation losses along with test metrics
    and save to the specified directory
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.colors import ListedColormap
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"{model_name}_loss.png"))
    plt.close()
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = test_metrics['confusion_matrix']
    
    # Create a custom colormap (white for 0, blue for higher values)
    cmap = ListedColormap(['white', 'lightskyblue', 'royalblue', 'navy'])
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Only 2 classes for Siamese networks (same/different)
    classes = ['Different', 'Same']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()
    
    # Plot distance distribution
    plt.figure(figsize=(10, 5))
    distances = np.array(test_metrics['distances'])
    labels = np.array(test_metrics['distances'])
    
    plt.hist([distances[labels==1], distances[labels==0]], bins=30, 
             label=['Same Class', 'Different Class'], alpha=0.6)
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.title('Distance Distribution')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"{model_name}_distance_distribution.png"))
    plt.close()

def save_siamese_metrics_to_excel(train_losses, val_losses, test_metrics, filename):
    """
    Save training, validation, and test metrics to an Excel file
    """
    import pandas as pd
    
    # Create DataFrames
    df_losses = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Training Loss': train_losses,
        'Validation Loss': val_losses
    })
    
    df_metrics = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [
            test_metrics['accuracy'],
            test_metrics['precision'],
            test_metrics['recall'],
            test_metrics['f1']
        ]
    })
    
    # Save to Excel
    with pd.ExcelWriter(filename) as writer:
        df_losses.to_excel(writer, sheet_name='Loss Curves', index=False)
        df_metrics.to_excel(writer, sheet_name='Test Metrics', index=False)

def main():
    data_dir = "fingerprint_data"
    
    # Prepare data
    train_loader, val_loader, test_loader, raw_test_loader, num_classes = prepare_dataloaders(data_dir)
    
    # Initialize Siamese network
    model = SiameseNet()
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS
    )
    
    # Find optimal threshold for binary classification
    optimal_threshold = find_optimal_threshold(model, val_loader)
    
    # Evaluate model using the optimal threshold
    test_metrics = evaluate_siamese_model(model, test_loader, threshold=optimal_threshold)
    
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1-Score: {test_metrics['f1']:.4f}")
    
    # Evaluate identification accuracy (1:N matching)
    id_accuracy = identification_accuracy(model, raw_test_loader)
    print(f"Identification Accuracy (1:N): {id_accuracy:.4f}")
    
    # Add identification accuracy to test metrics
    test_metrics['identification_accuracy'] = id_accuracy
    
    # Plot metrics
    plot_siamese_metrics(train_losses, val_losses, test_metrics, model_name=MODEL_NAME, save_dir=ASSETS_DIR)
    
    # Save metrics to Excel
    save_siamese_metrics_to_excel(train_losses, val_losses, test_metrics, 
                                 os.path.join(ASSETS_DIR, f"{MODEL_NAME}_evaluation.xlsx"))

if __name__ == "__main__":
    main()