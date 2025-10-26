# Group Assessment: Dog Breed Convolutional Neural Network
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn

# Epoch Function
def train_epoch(model, loader, criterion, optimizer, device):
    """Execute 1 training epoch: Returns loss & accuracy"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100.0 * correct / total

# Validate Function
def validate(model, loader, criterion, device):
    """Evaluates performance after every epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100.0 * correct / total

def test_model(model, loader, device):
    y_test = []
    y_prediction = []
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Predicted class index
            _, predicted = outputs.max(1)

            # Ground Truth Labels
            y_test.extend(labels.cpu().numpy())
            # Predicted Labels
            y_prediction.extend(predicted.cpu().numpy())

            # Calculate correct predictions
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    # Save Test Model's Classification Metrics
    test_accuracy = 100.0 * correct / total

    return test_accuracy, y_test, y_prediction

if __name__ == '__main__':
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create Local Path - UPDATE THIS TO YOUR PATH
    data_dir = r'Project\images'

    print(f"\n✅ Dataset path: {data_dir}")
    print(f"✅ Path exists: {os.path.exists(data_dir)}")

    # Define transforms FIRST
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load dataset WITHOUT transforms first (just for splitting)
    temp_dataset = datasets.ImageFolder(data_dir, transform=None)
    num_classes = len(temp_dataset.classes)
    cleaned_breed_labels = temp_dataset.classes  # Store class names

    print(f"✅ Dataset ready: {len(temp_dataset)} images, {num_classes} breeds")

    # Split into training (70%), validation (15%) & test sets (15%)
    train_size = int(0.7 * len(temp_dataset))
    val_size = int(0.15 * len(temp_dataset))
    test_size = len(temp_dataset) - train_size - val_size

    # Split the dataset
    train_indices, val_indices, test_indices = random_split(
        range(len(temp_dataset)),
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Now create datasets with appropriate transforms
    train_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(data_dir, transform=val_transform)

    # Create subsets using the indices
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=0)

    print(f"Train: {len(train_subset)} | Val: {len(val_subset)} | Test: {len(test_subset)}")

    # Create Residual Network (ResNet: 50 Layers)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze parameters for faster training
    for param in model.parameters():
        param.requires_grad = False

    # Replace last layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)  # Use dynamic number of classes
    )

    # DEBUG: CRITICAL CHECK
    print("\n=== MODEL ARCHITECTURE CHECK ===")
    print(f"Type of model.fc: {type(model.fc)}")
    print(f"model.fc contents:\n{model.fc}")

    # Check output size
    dummy_input = torch.randn(1, in_features)
    dummy_output = model.fc(dummy_input)
    print(f"\nOutput shape test: {dummy_output.shape}")
    print(f"Expected: torch.Size([1, {num_classes}])")

    if dummy_output.shape[1] != num_classes:
        print(f"❌ ERROR: Output is not {num_classes} classes!")
        exit(1)
    else:
        print("✅ Output shape correct!")
    print("================================\n")

    # Move model to device
    model = model.to(device)

    # Initialize Loss Function, Optimizer & Learning Rate Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Initialize number of epochs
    num_epochs = 10

    # Initialize empty lists for training/validation loss & accuracies
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    best_val_accuracy = 0.0

    # Training loop
    print("\n" + "="*50)
    print(" "*20 + "TRAINING" + " "*20)
    print("="*50)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        # Append losses & accuracies
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Feedback on performance
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

        # Save the model if validation accuracy improves
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_dog_breed_model.pth')
            print(f"✅ Saved best model: {val_accuracy:.2f}%")

        scheduler.step()

    # Overall best performance
    print(f"\n{'='*50}")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
    print(f"{'='*50}\n")

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train', marker='o')
    plt.plot(val_losses, label='Validation', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train', marker='o')
    plt.plot(val_accuracies, label='Validation', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy over Epochs')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Testing phase
    print("\n" + "="*50)
    print(" "*20 + "TESTING" + " "*20)
    print("="*50)

    try:
        model.load_state_dict(torch.load('best_dog_breed_model.pth'))
        print("✅ Successfully loaded best model for testing\n")
    except FileNotFoundError:
        print("⚠️ Warning: 'best_dog_breed_model.pth' not found.")
        print("Using the model from the last training epoch\n")

    # Run final test
    final_test_accuracy, y_test, y_prediction = test_model(model, test_loader, device)

    print(f"\n{'='*50}")
    print(f"Final Test Accuracy: {final_test_accuracy:.2f}%")
    print(f"{'='*50}\n")

    # Classification Report
    print("="*60)
    print(" "*20 + "CLASSIFICATION REPORT" + " "*20)
    print("="*60)

    print(classification_report(
        y_test,
        y_prediction,
        target_names=cleaned_breed_labels[:num_classes],  # Ensure we use correct number of labels
        zero_division=0,
        digits=3
    ))

    # Confusion Matrix Visualization
    print("\n" + "="*50)
    print(" "*15 + "GENERATING CONFUSION MATRICES" + " "*15)
    print("="*50)

    # Visualize confusion matrices in groups (if more than 30 classes)
    total_classes = num_classes

    if total_classes <= 30:
        # Single confusion matrix for small number of classes
        cm = confusion_matrix(y_test, y_prediction)

        plt.figure(figsize=(12, 10))
        sn.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=cleaned_breed_labels[:total_classes],
            yticklabels=cleaned_breed_labels[:total_classes],
            annot_kws={"size": 8}
        )
        plt.title('Confusion Matrix - All Classes', fontsize=16)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix_all.png', dpi=150, bbox_inches='tight')
        plt.show()
    else:
        # Multiple confusion matrices for many classes
        num_splits = 4
        classes_per_split = total_classes // num_splits

        y_test_np = np.array(y_test)
        y_prediction_np = np.array(y_prediction)

        for i in range(num_splits):
            start_index = i * classes_per_split
            end_index = min((i + 1) * classes_per_split, total_classes)

            # Create mask for current subset of classes
            mask = np.logical_and(
                y_test_np >= start_index,
                y_test_np < end_index
            )

            if not np.any(mask):
                continue

            y_test_subset = y_test_np[mask]
            y_prediction_subset = y_prediction_np[mask]

            # Adjust indices for confusion matrix
            y_test_adjusted = y_test_subset - start_index
            y_prediction_adjusted = y_prediction_subset - start_index

            current_labels = cleaned_breed_labels[start_index:end_index]

            cm = confusion_matrix(
                y_test_adjusted,
                y_prediction_adjusted,
                labels=range(len(current_labels))
            )

            plt.figure(figsize=(10, 8))
            sn.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=current_labels,
                yticklabels=current_labels,
                annot_kws={"size": 6}
            )

            plt.title(f'Confusion Matrix (Classes {start_index} to {end_index-1})', fontsize=14)
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(f'confusion_matrix_part_{i+1}.png', dpi=150, bbox_inches='tight')
            print(f"✅ Matrix {i+1} of {num_splits} generated (Classes {start_index} to {end_index-1})")
            plt.show()

    print("\n✅ All visualizations complete!")

    print(f"Model and results saved in current directory.")
