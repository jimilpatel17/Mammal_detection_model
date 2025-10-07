import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

# =========================
# Configuration
# =========================
class Config:
    # Paths (CHANGE THESE to your dataset folders)
    mammal_dir = "/home/aidev/projects/my_model/Dataset/data_splitted/val"
    non_mammal_dir = "/home/aidev/projects/my_model/not_a_mammal"
    model_save_path = "/home/aidev/projects/my_model/Dataset/CHECKPOINTS/mammal_detector.pth"

    # Training parameters
    batch_size = 32
    num_workers = 4
    lr = 0.001
    epochs = 50
    img_size = 224

    # Model architecture
    model_name = "efficientnet_b0"
    pretrained = True

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Custom dataset
# =========================
class MammalDataset(Dataset):
    def __init__(self, mammal_dir, non_mammal_dir, transform=None):
        self.transform = transform
        valid_exts = (".jpg", ".jpeg", ".png", ".webp")

        # ---- Mammals (scan subfolders) ----
        all_mammal_paths = []
        species_counts = []
        for root, _, files in os.walk(mammal_dir):
            images = [
                os.path.join(root, f) for f in files if f.lower().endswith(valid_exts)
            ]
            if images:
                species_counts.append(len(images))
                all_mammal_paths.extend(images)

        # Balance mammal species if we found any
        if species_counts:
            median_count = int(np.median(species_counts))
            sampled_indices = np.random.choice(
                len(all_mammal_paths),
                size=median_count * len(species_counts),
                replace=True,
            )
            self.mammal_images = [(all_mammal_paths[i], 1) for i in sampled_indices]
        else:
            self.mammal_images = []

        # ---- Non-mammals (scan subfolders) ----
        non_mammal_files = []
        for root, _, files in os.walk(non_mammal_dir):
            non_mammal_files.extend(
                os.path.join(root, f) for f in files if f.lower().endswith(valid_exts)
            )
        self.non_mammal_images = [(f, 0) for f in non_mammal_files]

        # ---- Balance classes ----
        n_samples = min(len(self.mammal_images), len(self.non_mammal_images))
        self.mammal_images = self.mammal_images[:n_samples]
        self.non_mammal_images = self.non_mammal_images[:n_samples]

        self.all_images = self.mammal_images + self.non_mammal_images
        np.random.shuffle(self.all_images)

        print(
            f"[DEBUG] Loaded {len(self.mammal_images)} mammal and {len(self.non_mammal_images)} non-mammal samples"
        )

        if len(self.all_images) == 0:
            raise ValueError("Dataset is empty! Check your paths and images.")

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path, label = self.all_images[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # fallback: return a random other sample
            return self[np.random.randint(len(self))]


# =========================
# Data transformations     
# =========================
train_transform = transforms.Compose([
    transforms.Resize((Config.img_size, Config.img_size)),
    transforms.RandomResizedCrop(Config.img_size, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)),
])

val_transform = transforms.Compose([
    transforms.Resize((Config.img_size, Config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# =========================
# Model
# =========================
def create_model():
    model = models.efficientnet_b0(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(1280, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 1),
    )
    return model.to(Config.device)


# =========================
# Training loop
# =========================
def train():
    dataset = MammalDataset(Config.mammal_dir, Config.non_mammal_dir, train_transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers
    )

    model = create_model()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    best_val_acc = 0.0
    patience, wait = 3, 0

    for epoch in range(Config.epochs):
        # Training
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.epochs} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(Config.device), labels.float().to(Config.device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(
                {"loss": running_loss / (total / Config.batch_size), "acc": correct / total}
            )

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{Config.epochs} [Val]")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(Config.device), labels.float().to(Config.device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                pbar.set_postfix(
                    {
                        "val_loss": val_loss / (val_total / Config.batch_size),
                        "val_acc": val_correct / val_total,
                    }
                )

        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc, wait = val_acc, 0
            torch.save(model.state_dict(), Config.model_save_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break


# =========================
# Inference
# =========================
class MammalDetector:
    def __init__(self, model_path=Config.model_save_path):
        self.model = create_model()
        self.model.load_state_dict(torch.load(model_path, map_location=Config.device))
        self.model.eval()
        self.transform = val_transform

    def predict(self, image_path, threshold=0.8):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(Config.device)
        with torch.no_grad():
            prob = torch.sigmoid(self.model(image)).item()
        return {
            "is_mammal": prob > threshold,
            "confidence": prob,
            "message": "Mammal detected!" if prob > threshold else "Not a mammal.",
        }


if __name__ == "__main__":
    train()
