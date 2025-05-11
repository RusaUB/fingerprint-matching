import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image, ImageOps
import random
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

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
NUM_EPOCHS = 1
IMAGE_SIZE = (288, 384)
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

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
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, full_dataset.get_num_classes()

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        
        # Define a simple CNN with increasing filter sizes
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth convolutional block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fifth convolutional block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # Global average pooling to 1x1
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layer for classification
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Pass through convolutional layers
        x = self.features(x)
        
        # Flatten the features
        x = x.view(x.size(0), -1)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def get_features(self, x):
        # Extract feature vectors before the classifier
        x = self.features(x)
        return x.view(x.size(0), -1)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, '
              f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')
        
        # Save the best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), os.path.join(CURRENT_DIR, f"{MODEL_NAME}_model.pth"))
    
    # Load the best model
    model.load_state_dict(torch.load(os.path.join(CURRENT_DIR, f"{MODEL_NAME}_model.pth")))
    
    return model, train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate overall accuracy
    accuracy = correct / total
    
    # Calculate precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def compute_similarity_matching(model, test_loader, threshold=0.8):
    """
    Evaluate the model's ability to verify if two fingerprints belong to the same person
    by comparing feature similarity rather than direct classification
    """
    model.eval()
    
    # Create pairs of images from test set - both same class and different class
    # For simplicity, we'll use the existing test_loader and create pairs on-the-fly
    
    all_features = []
    all_labels = []
    
    # Extract features and labels from all test images
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            features = model.get_features(images)
            
            all_features.append(features.cpu())
            all_labels.append(labels)
    
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Generate pairs
    num_samples = len(all_features)
    max_pairs = min(1000, num_samples * 2)  # Limit number of pairs to evaluate
    
    same_class_correct = 0
    same_class_total = 0
    diff_class_correct = 0
    diff_class_total = 0
    
    for _ in range(max_pairs):
        # Randomly select two indices
        idx1 = random.randint(0, num_samples - 1)
        idx2 = random.randint(0, num_samples - 1)
        
        if idx1 == idx2:
            continue
        
        # Get features and compute cosine similarity
        f1 = all_features[idx1]
        f2 = all_features[idx2]
        
        # Normalize features for cosine similarity
        f1_norm = f1 / f1.norm()
        f2_norm = f2 / f2.norm()
        similarity = torch.dot(f1_norm, f2_norm).item()
        
        # Check if same class
        same_class = (all_labels[idx1] == all_labels[idx2]).item()
        
        # For same class pairs
        if same_class:
            same_class_total += 1
            if similarity >= threshold:
                same_class_correct += 1
        else:
            diff_class_total += 1
            if similarity < threshold:
                diff_class_correct += 1
    
    # Calculate accuracy for same class and different class pairs
    same_class_acc = same_class_correct / same_class_total if same_class_total > 0 else 0
    diff_class_acc = diff_class_correct / diff_class_total if diff_class_total > 0 else 0
    
    # Overall verification accuracy
    overall_acc = (same_class_correct + diff_class_correct) / (same_class_total + diff_class_total)
    
    print(f"Similarity Matching Results:")
    print(f"  Same Class Accuracy: {same_class_acc:.4f} ({same_class_correct}/{same_class_total})")
    print(f"  Different Class Accuracy: {diff_class_acc:.4f} ({diff_class_correct}/{diff_class_total})")
    print(f"  Overall Verification Accuracy: {overall_acc:.4f}")
    
    return {
        'same_class_acc': same_class_acc,
        'diff_class_acc': diff_class_acc,
        'overall_acc': overall_acc
    }

def main():
    data_dir = "fingerprint_data"
    
    # Prepare data
    train_loader, val_loader, test_loader, num_classes = prepare_dataloaders(data_dir)
    
    # Initialize model
    model = SimpleCNN(num_classes)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS
    )
    
    # Evaluate model
    test_metrics = evaluate_model(model, test_loader)
    
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1-Score: {test_metrics['f1']:.4f}")
    
    # Also evaluate using feature similarity for verification
    verification_metrics = compute_similarity_matching(model, test_loader)
    
    # Add verification metrics to test metrics
    test_metrics.update(verification_metrics)
    
    # Plot metrics
    plot_metrics(train_losses, val_losses, train_accs, val_accs, test_metrics, model_name=MODEL_NAME, save_dir=ASSETS_DIR)

    save_metrics_to_excel(train_losses, val_losses, train_accs, val_accs, test_metrics, os.path.join(ASSETS_DIR, f"{MODEL_NAME}_evaluation.xlsx"))

if __name__ == "__main__":
    main()