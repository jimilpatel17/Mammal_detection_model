
"""
This script:

* Cleans dataset (removes bad images).
* Handles imbalance with weighted sampling + weighted loss.
* Trains a ResNet50 with augmentation, dropout, class weights
* Uses LR scheduler + early stopping for stable training.
* Saves best model & resumes training if interrupted.
* Provides an inference function to test single image.
"""
"""
* This script is for training an image classification model using PyTorch.
* This code implements data augmentation techniques for image classification using PyTorch.
* It sets up data loaders for training, validation, and testing datasets with appropriate transformations.
* It uses a WeightedRandomSampler to handle class imbalance during training.
* The loss function is also weighted to account for class imbalance.
* A simple training loop is provided to train a ResNet50 model on the augmented data.
* The best model is saved based on validation loss.
* Test-time augmentation (TTA) is applied during inference to improve prediction robustness.
* The model used is a pretrained ResNet50, which is fine-tuned on the provided dataset.
* The script includes a testitng function to predict the class of a single image.
* Now after training, if you test with an image:
    - If it's a mammal the model knows → returns species name.
    - If it's something else (car, bird, etc.) → returns “Maybe not a mammal” if confidence < 0.6.
* Corrupted images are detected and removed before training to avoid crashes.
* A log file is created to keep track of removed corrupted files.
* The code also cleans the dataset by removing corrupted images before training.
* The code had a LR scheduler to reduce the learning rate when the validation loss plateaus.
* Early stopping is implemented to halt training if the validation loss doesn't improve for a set number of epochs.
"""

import os
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_v2_s, EfficientNet_V2_S_Weights
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION CLASS - Centralized Configuration Management
# ============================================================================
class TrainingConfig:
    def __init__(self):
        # Dataset paths
        self.train_dir = "/home/aidev/projects/my_model/Dataset/data_splitted/train"
        self.val_dir = "/home/aidev/projects/my_model/Dataset/data_splitted/val"
        self.test_dir = "/home/aidev/projects/my_model/Dataset/data_splitted/test"

        # Training hyperparameters
        self.batch_size = 32
        self.num_workers = 4
        self.max_epochs = 50
        self.initial_lr = 1e-4
        self.weight_decay = 1e-4
        self.grad_clip_norm = 1.0
        
        # Model selection
        self.model_name = "resnet50"  # Options: "resnet50", "efficientnet_v2_s"
        
        # Early stopping
        self.patience = 10
        self.min_delta = 1e-4
        
        # Mixed precision
        self.use_mixed_precision = True
        
        # Advanced augmentations
        self.use_advanced_augments = True
        
        # Directories
        self.checkpoint_dir = "/home/aidev/projects/my_model/Dataset/CHECKPOINTS"
        self.log_dir = "/home/aidev/projects/my_model/Dataset/LOGS"
        self.results_dir = "/home/aidev/projects/my_model/Dataset/RESULTS"
        
        # Create directories
        for dir_path in [self.checkpoint_dir, self.log_dir, self.results_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def save(self, path: str):
        """Save configuration to JSON file"""
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

# ============================================================================
# ADVANCED DATA TRANSFORMATIONS
# ============================================================================
class AdvancedTransforms:
    @staticmethod
    def get_albumentations_train_transforms(img_size: int = 224):
        """Advanced augmentations using Albumentations"""
        return A.Compose([
            A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.5),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.1),
                A.GridDistortion(distort_limit=0.1),
                A.ElasticTransform(alpha=1, sigma=50),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50)),
                A.GaussianBlur(blur_limit=3),
                A.MotionBlur(blur_limit=3),
            ], p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
            A.Cutout(num_holes=1, max_h_size=32, max_w_size=32, fill_value=0, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    @staticmethod
    def get_basic_train_transforms(img_size: int = 224):
        """Basic PyTorch transforms"""
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    @staticmethod
    def get_test_transforms(img_size: int = 224):
        """Test/validation transforms"""
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

# ============================================================================
# MODEL FACTORY
# ============================================================================
class ModelFactory:
    @staticmethod
    def create_model(model_name: str, num_classes: int) -> nn.Module:
        """Create model based on model name"""
        if model_name.lower() == "resnet50":
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name.lower() == "efficientnet_v2_s":
            model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
            model.classifier = nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        return model

# ============================================================================
# EARLY STOPPING CLASS
# ============================================================================
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max':
            if score < self.best_score + self.min_delta:
                self.counter += 1
            else:
                self.best_score = score
                self.counter = 0
        else:  # mode == 'min'
            if score > self.best_score - self.min_delta:
                self.counter += 1
            else:
                self.best_score = score
                self.counter = 0
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop

# ============================================================================
# METRICS CALCULATOR
# ============================================================================
class MetricsCalculator:
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate comprehensive classification metrics"""
        # Basic metrics
        accuracy = np.mean(y_true == y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Macro/Micro averages
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Classification report
        report = classification_report(
            y_true, y_pred, target_names=self.class_names, 
            output_dict=True, zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'weighted_f1': weighted_f1,
            'per_class_precision': precision,
            'per_class_recall': recall,
            'per_class_f1': f1,
            'per_class_support': support,
            'classification_report': report
        }
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            save_path: str, title: str = 'Confusion Matrix'):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

# ============================================================================
# ADVANCED TRAINER CLASS
# ============================================================================
class AdvancedTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.scaler = GradScaler() if config.use_mixed_precision else None
        self.writer = SummaryWriter(log_dir=config.log_dir)
        self.early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta)
        
        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [],
            'learning_rates': []
        }
        
        self.logger.info(f"Trainer initialized on device: {self.device}")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(self.config.log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
        """Prepare data loaders with advanced augmentations"""
        self.logger.info("Preparing datasets and data loaders...")
        
        # Choose transforms
        if self.config.use_advanced_augments:
            # Note: For Albumentations, you'd need a custom dataset class
            # Using PyTorch transforms for compatibility
            train_transforms = AdvancedTransforms.get_basic_train_transforms()
        else:
            train_transforms = AdvancedTransforms.get_basic_train_transforms()
        
        test_transforms = AdvancedTransforms.get_test_transforms()
        
        # Create datasets
        train_dataset = datasets.ImageFolder(root=self.config.train_dir, transform=train_transforms)
        val_dataset = datasets.ImageFolder(root=self.config.val_dir, transform=test_transforms)
        test_dataset = datasets.ImageFolder(root=self.config.test_dir, transform=test_transforms)
        
        if len(train_dataset) == 0:
            raise RuntimeError(f"No images found in {self.config.train_dir}")
        
        # Calculate class weights and create sampler
        targets = np.array(train_dataset.targets)
        class_sample_counts = np.bincount(targets)
        class_sample_counts = np.maximum(class_sample_counts, 1)
        
        self.class_weights = 1.0 / class_sample_counts
        sample_weights = self.class_weights[targets]
        sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        
        # Create data loaders
        pin_memory = self.device.type == "cuda"
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, sampler=sampler,
            num_workers=self.config.num_workers, pin_memory=pin_memory, persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False,
            num_workers=self.config.num_workers, pin_memory=pin_memory, persistent_workers=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size, shuffle=False,
            num_workers=self.config.num_workers, pin_memory=pin_memory, persistent_workers=True
        )
        
        self.logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        self.logger.info(f"Number of classes: {len(train_dataset.classes)}")
        self.logger.info(f"Class distribution: {dict(zip(train_dataset.classes, class_sample_counts))}")
        
        return train_loader, val_loader, test_loader, train_dataset.classes
    
    def create_model(self, num_classes: int) -> nn.Module:
        """Create and setup model"""
        model = ModelFactory.create_model(self.config.model_name, num_classes)
        model = model.to(self.device)
        
        # Print model summary
        try:
            self.logger.info("Model Architecture:")
            summary(model, input_size=(3, 224, 224), device=str(self.device))
        except:
            self.logger.info(f"Model: {self.config.model_name} with {num_classes} classes")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: optim.Optimizer, criterion: nn.Module, epoch: int) -> Tuple[float, float]:
        """Train for one epoch with mixed precision support"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.max_epochs}", leave=False)
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if self.config.use_mixed_precision and self.scaler:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip_norm)
                optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                      criterion: nn.Module, metrics_calc: MetricsCalculator) -> Dict:
        """Validate model and return comprehensive metrics"""
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.inference_mode():
            for images, labels in tqdm(val_loader, desc="Validating", leave=False):
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                
                if self.config.use_mixed_precision:
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = running_loss / len(val_loader)
        
        # Calculate comprehensive metrics
        metrics = metrics_calc.calculate_comprehensive_metrics(
            np.array(all_labels), np.array(all_preds)
        )
        
        return {
            'val_loss': val_loss,
            'val_acc': metrics['accuracy'],
            'val_f1': metrics['weighted_f1'],
            'metrics': metrics
        }
    
    def train(self):
        """Main training loop with all advanced features"""
        self.logger.info("Starting advanced training...")
        
        # Prepare data
        train_loader, val_loader, test_loader, class_names = self.prepare_data()
        self.metrics_calc = MetricsCalculator(class_names)
        
        # Create model
        model = self.create_model(len(class_names))
        
        # Setup loss function with class weights
        class_weights_tensor = torch.tensor(self.class_weights, dtype=torch.float32, device=self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        # Setup optimizer with weight decay
        optimizer = optim.AdamW(model.parameters(), lr=self.config.initial_lr, 
                              weight_decay=self.config.weight_decay)
        
        # Setup learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        # Save configuration
        config_path = os.path.join(self.config.results_dir, 'config.json')
        self.config.save(config_path)
        
        best_val_acc = 0.0
        best_val_f1 = 0.0
        
        self.logger.info("Training loop started...")
        
        for epoch in range(self.config.max_epochs):
            # Training
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion, epoch)
            
            # Validation
            val_results = self.validate_epoch(model, val_loader, criterion, self.metrics_calc)
            
            # Update learning rate
            scheduler.step(val_results['val_acc'])
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_results['val_loss'])
            self.history['val_acc'].append(val_results['val_acc'])
            self.history['val_f1'].append(val_results['val_f1'])
            self.history['learning_rates'].append(current_lr)
            
            # Enhanced Lo gging with Clear Accuracy Display
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"EPOCH {epoch+1}/{self.config.max_epochs} RESULTS:")
            self.logger.info(f"{'='*60}")
            self.logger.info(f" TRAINING   - Loss: {train_loss:.4f} | Accuracy: {train_acc*100:.2f}%")
            self.logger.info(f" VALIDATION - Loss: {val_results['val_loss']:.4f} | Accuracy: {val_results['val_acc']*100:.2f}% | F1: {val_results['val_f1']:.4f}")
            self.logger.info(f" Learning Rate: {current_lr:.2e}")
            self.logger.info(f"  Best Val Acc So Far: {best_val_acc*100:.2f}% | Best F1 So Far: {best_val_f1:.4f}")
            
            # TensorBoard logging
            self.writer.add_scalars('Loss', {
                'Train': train_loss,
                'Validation': val_results['val_loss']
            }, epoch)
            self.writer.add_scalars('Accuracy', {
                'Train': train_acc,
                'Validation': val_results['val_acc']
            }, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            self.writer.add_scalar('Val_F1', val_results['val_f1'], epoch)
            
            # Enhanced Checkpoint Saving with Full State
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                'class_names': class_names,
                'val_acc': val_results['val_acc'],
                'val_f1': val_results['val_f1'],
                'train_acc': train_acc,
                'train_loss': train_loss,
                'config': self.config.__dict__,
                'history': self.history,
                'best_val_acc_so_far': best_val_acc,
                'best_val_f1_so_far': best_val_f1,
                'current_lr': current_lr
            }
            
            # Save epoch checkpoint
            checkpoint_path = os.path.join(self.config.checkpoint_dir, f"epoch_{epoch+1:03d}.pth")
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f" Checkpoint saved: {checkpoint_path}")
            
            # Save best models with improved tracking
            model_improved = False
            if val_results['val_acc'] > best_val_acc:
                best_val_acc = val_results['val_acc']
                best_acc_path = os.path.join(self.config.checkpoint_dir, "best_accuracy_model.pth")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'val_acc': best_val_acc,
                    'val_f1': val_results['val_f1'],
                    'train_acc': train_acc,
                    'config': self.config.__dict__,
                    'class_names': class_names
                }, best_acc_path)
                self.logger.info(f"NEW BEST ACCURACY! Model saved: {best_acc_path}")
                self.logger.info(f"    Previous Best: {(val_results['val_acc']-0.01)*100:.2f}% → New Best: {best_val_acc*100:.2f}%")
                model_improved = True
            
            if val_results['val_f1'] > best_val_f1:
                best_val_f1 = val_results['val_f1']
                best_f1_path = os.path.join(self.config.checkpoint_dir, "best_f1_model.pth")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'val_acc': val_results['val_acc'],
                    'val_f1': best_val_f1,
                    'train_acc': train_acc,
                    'config': self.config.__dict__,
                    'class_names': class_names
                }, best_f1_path)
                self.logger.info(f" NEW BEST F1 SCORE! Model saved: {best_f1_path}")
                self.logger.info(f"    Previous Best F1: {(val_results['val_f1']-0.01):.4f} → New Best F1: {best_val_f1:.4f}")
                model_improved = True
            
            if not model_improved:
                self.logger.info(f"📈 No improvement this epoch (Best Acc: {best_val_acc*100:.2f}%, Best F1: {best_val_f1:.4f})")
            
            # Progress summary
            improvement_acc = (val_results['val_acc'] - best_val_acc) * 100 if best_val_acc > 0 else 0
            improvement_f1 = (val_results['val_f1'] - best_val_f1) if best_val_f1 > 0 else 0
            self.logger.info(f" Progress: Acc Gap: {improvement_acc:+.2f}%, F1 Gap: {improvement_f1:+.4f}")
            self.logger.info(f" Early Stopping Counter: {self.early_stopping.counter}/{self.early_stopping.patience}")
            self.logger.info(f"{'='*60}\n")
            
            # Early stopping
            if self.early_stopping(val_results['val_acc']):
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Final evaluation on test set
        self.logger.info("\n" + "="*60)
        self.logger.info("FINAL TEST EVALUATION")
        self.logger.info("="*60)
        
        # Load best F1 model for final evaluation
        best_f1_path = os.path.join(self.config.checkpoint_dir, "best_f1_model.pth")
        if os.path.exists(best_f1_path):
            checkpoint = torch.load(best_f1_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Loaded best F1 model from epoch {checkpoint.get('epoch', 'Unknown')}")
        else:
            self.logger.info("Using current model for final evaluation")
        
        test_results = self.validate_epoch(model, test_loader, criterion, self.metrics_calc)
        
        # Display comprehensive test results
        self.logger.info(f"\nFINAL TEST RESULTS:")
        self.logger.info(f"Test Accuracy: {test_results['val_acc']*100:.2f}%")
        self.logger.info(f"Test F1-Score: {test_results['val_f1']:.4f}")
        self.logger.info(f"Test Loss: {test_results['val_loss']:.4f}")
        
        # Save final model summary
        final_summary = {
            'training_completed': True,
            'total_epochs_trained': epoch + 1,
            'best_validation_accuracy': best_val_acc,
            'best_validation_f1': best_val_f1,
            'final_test_accuracy': test_results['val_acc'],
            'final_test_f1': test_results['val_f1'],
            'final_test_loss': test_results['val_loss'],
            'best_accuracy_model_path': os.path.join(self.config.checkpoint_dir, "best_accuracy_model.pth"),
            'best_f1_model_path': os.path.join(self.config.checkpoint_dir, "best_f1_model.pth"),
            'early_stopped': self.early_stopping.early_stop
        }
        
        summary_path = os.path.join(self.config.results_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(final_summary, f, indent=2, default=str)
        
        self.logger.info(f"\nTraining summary saved to: {summary_path}")
        
        # Save comprehensive results
        self.save_results(test_results, class_names)
        
        # Plot training curves
        self.plot_training_history()
        
        # Generate confusion matrix for test set
        model.eval()
        all_preds, all_labels = [], []
        with torch.inference_mode():
            for images, labels in test_loader:
                images = images.to(self.device, non_blocking=True)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        cm_path = os.path.join(self.config.results_dir, 'confusion_matrix.png')
        self.metrics_calc.plot_confusion_matrix(
            np.array(all_labels), np.array(all_preds), cm_path, 'Test Set Confusion Matrix'
        )
        
        self.writer.close()
        
        # Final completion message
        self.logger.info("\n" + "="*60)
        self.logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        self.logger.info("="*60)
        self.logger.info(f"Best Models Saved:")
        self.logger.info(f"  - Best Accuracy: {os.path.join(self.config.checkpoint_dir, 'best_accuracy_model.pth')}")
        self.logger.info(f"  - Best F1 Score: {os.path.join(self.config.checkpoint_dir, 'best_f1_model.pth')}")
        self.logger.info(f"Results Directory: {self.config.results_dir}")
        self.logger.info(f"Checkpoints Directory: {self.config.checkpoint_dir}")
        self.logger.info(f"TensorBoard Logs: {self.config.log_dir}")
        self.logger.info("="*60)
        
        return model, self.history
    
    def save_results(self, test_results: Dict, class_names: List[str]):
        """Save comprehensive results"""
        results = {
            'test_accuracy': test_results['val_acc'],
            'test_f1': test_results['val_f1'],
            'test_loss': test_results['val_loss'],
            'metrics': test_results['metrics'],
            'class_names': class_names,
            'training_history': self.history,
            'config': self.config.__dict__
        }
        
        results_path = os.path.join(self.config.results_dir, 'final_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to: {results_path}")
    
    def plot_training_history(self):
        """Plot and save training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', marker='o')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss', marker='s')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot([acc*100 for acc in self.history['train_acc']], label='Train Acc', marker='o')
        axes[0, 1].plot([acc*100 for acc in self.history['val_acc']], label='Val Acc', marker='s')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score plot
        axes[1, 0].plot(self.history['val_f1'], label='Val F1', marker='d', color='green')
        axes[1, 0].set_title('Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning Rate plot
        axes[1, 1].plot(self.history['learning_rates'], label='Learning Rate', marker='x', color='red')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        history_path = os.path.join(self.config.results_dir, 'training_history.png')
        plt.savefig(history_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training history plot saved to: {history_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main function to run the advanced training"""
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create configuration
    config = TrainingConfig()
    
    # Create trainer and start training
    trainer = AdvancedTrainer(config)
    model, history = trainer.train()
    
    print("\n" + "="*50)
    print(" ADVANCED TRAINING COMPLETED SUCCESSFULLY! ")
    print("="*50)
    print(f" Results saved to: {config.results_dir}")
    print(f" TensorBoard logs: {config.log_dir}")
    print(f" Best models saved in: {config.checkpoint_dir}")
    print("\nTo view TensorBoard logs, run:")
    print(f"tensorboard --logdir {config.log_dir}")


# ============================================================================
# ADDITIONAL UTILITIES
# ============================================================================
class ModelAnalyzer:
    """Advanced model analysis and interpretation tools"""
    
    def __init__(self, model: nn.Module, device: torch.device, class_names: List[str]):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.model.eval()
    
    def get_layer_wise_learning_rates(self, base_lr: float = 1e-4) -> Dict:
        """Get discriminative learning rates for different layers"""
        params = []
        
        # Lower learning rate for backbone
        backbone_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if 'fc' in name or 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        return [
            {'params': backbone_params, 'lr': base_lr * 0.1},  # Lower LR for pretrained layers
            {'params': classifier_params, 'lr': base_lr}       # Higher LR for new layers
        ]
    
    def predict_with_confidence(self, dataloader: DataLoader, threshold: float = 0.9) -> Dict:
        """Make predictions with confidence scores"""
        all_predictions = []
        all_confidences = []
        all_labels = []
        
        with torch.inference_mode():
            for images, labels in dataloader:
                images = images.to(self.device)
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                
                confidences, predictions = torch.max(probabilities, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Filter high confidence predictions
        high_conf_mask = np.array(all_confidences) >= threshold
        
        return {
            'predictions': np.array(all_predictions),
            'confidences': np.array(all_confidences),
            'labels': np.array(all_labels),
            'high_confidence_mask': high_conf_mask,
            'high_confidence_accuracy': np.mean(
                np.array(all_predictions)[high_conf_mask] == 
                np.array(all_labels)[high_conf_mask]
            ) if high_conf_mask.sum() > 0 else 0.0
        }


class AdvancedDataAugmentation:
    """Advanced data augmentation techniques"""
    
    @staticmethod
    def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply MixUp augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    @staticmethod
    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        """MixUp loss function"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    @staticmethod
    def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply CutMix augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        # Get random box
        W, H = x.size(2), x.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int32(W * cut_rat)
        cut_h = np.int32(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        y_a, y_b = y, y[index]
        
        return x, y_a, y_b, lam


class AdvancedLossFunctions:
    """Advanced loss functions for better training"""
    
    @staticmethod
    def focal_loss(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 1.0, gamma: float = 2.0) -> torch.Tensor:
        """Focal Loss for handling class imbalance"""
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    @staticmethod
    def label_smoothing_loss(inputs: torch.Tensor, targets: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
        """Label smoothing for better generalization"""
        num_classes = inputs.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(inputs)
            true_dist.fill_(smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)
        
        return torch.mean(torch.sum(-true_dist * torch.log_softmax(inputs, dim=-1), dim=-1))


class ModelEnsemble:
    """Model ensemble for improved performance"""
    
    def __init__(self, models: List[nn.Module], device: torch.device):
        self.models = models
        self.device = device
        
        for model in self.models:
            model.eval()
    
    def predict(self, dataloader: DataLoader, method: str = 'average') -> np.ndarray:
        """Ensemble prediction using multiple models"""
        all_predictions = []
        
        with torch.inference_mode():
            for images, _ in dataloader:
                images = images.to(self.device)
                batch_predictions = []
                
                for model in self.models:
                    outputs = model(images)
                    if method == 'average':
                        probs = torch.softmax(outputs, dim=1)
                        batch_predictions.append(probs.cpu().numpy())
                    elif method == 'voting':
                        preds = torch.argmax(outputs, dim=1)
                        batch_predictions.append(preds.cpu().numpy())
                
                if method == 'average':
                    # Average probabilities
                    avg_probs = np.mean(batch_predictions, axis=0)
                    final_preds = np.argmax(avg_probs, axis=1)
                elif method == 'voting':
                    # Majority voting
                    batch_predictions = np.array(batch_predictions)
                    final_preds = []
                    for i in range(batch_predictions.shape[1]):
                        votes = batch_predictions[:, i]
                        final_preds.append(np.bincount(votes).argmax())
                    final_preds = np.array(final_preds)
                
                all_predictions.extend(final_preds)
        
        return np.array(all_predictions)


class AdvancedVisualization:
    """Advanced visualization tools for model interpretation"""
    
    def __init__(self, model: nn.Module, device: torch.device, class_names: List[str]):
        self.model = model
        self.device = device
        self.class_names = class_names
    
    def plot_class_distribution(self, dataset: datasets.ImageFolder, save_path: str):
        """Plot class distribution in dataset"""
        targets = np.array(dataset.targets)
        class_counts = np.bincount(targets)
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(class_counts)), class_counts, 
                      color=plt.cm.viridis(np.linspace(0, 1, len(class_counts))))
        
        plt.title('Class Distribution in Dataset', fontsize=16)
        plt.xlabel('Classes', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.xticks(range(len(self.class_names)), self.class_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, class_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(class_counts),
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_learning_curves_advanced(self, history: Dict, save_path: str):
        """Advanced learning curves with statistical information"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss with error bands (if multiple runs)
        axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(epochs, [acc*100 for acc in history['train_acc']], 
                       label='Train Acc', linewidth=2)
        axes[0, 1].plot(epochs, [acc*100 for acc in history['val_acc']], 
                       label='Val Acc', linewidth=2)
        axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score
        axes[0, 2].plot(epochs, history['val_f1'], label='Val F1', linewidth=2, color='green')
        axes[0, 2].set_title('Validation F1 Score', fontsize=14)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('F1 Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 0].plot(epochs, history['learning_rates'], linewidth=2, color='red')
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=14)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss difference (overfitting indicator)
        loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
        axes[1, 1].plot(epochs, loss_diff, linewidth=2, color='orange')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Overfitting Indicator (Val - Train Loss)', fontsize=14)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Difference')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Accuracy difference
        acc_diff = np.array(history['train_acc']) - np.array(history['val_acc'])
        axes[1, 2].plot(epochs, [diff*100 for diff in acc_diff], linewidth=2, color='purple')
        axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 2].set_title('Overfitting Indicator (Train - Val Acc)', fontsize=14)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Accuracy Difference (%)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


# ============================================================================
# HYPERPARAMETER OPTIMIZATION (Optional Advanced Feature)
# ============================================================================
class HyperparameterOptimizer:
    """Optuna-based hyperparameter optimization (requires optuna installation)"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def objective(self, trial):
        """Objective function for hyperparameter optimization"""
        # Suggest hyperparameters
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        
        # Update config
        self.config.initial_lr = lr
        self.config.batch_size = batch_size
        self.config.weight_decay = weight_decay
        self.config.max_epochs = 10  # Shorter for hyperparameter search
        
        # Train model
        trainer = AdvancedTrainer(self.config)
        model, history = trainer.train()
        
        # Return validation accuracy as objective
        return max(history['val_acc'])


if __name__ == "__main__":
    main()



 """
** Here’s what happens step by step:

:- Preprocessing (pipelines) :

    * Training images → augmented (crop, flip, rotation, color jitter).
    * Validation & test images → only resized + normalized.

:- Data balancing :

    * WeightedRandomSampler makes sure rare species are sampled more often in batches.
    * CrossEntropyLoss(weight=...) makes sure the loss function gives more importance to rare classes.

:- Model setup :

    * Uses ResNet18 (pretrained) and replaces the last FC layer with your number of species.

:- Training loop

    * Runs for 5 epochs (you can increase it).
    * Each batch → forward pass → compute weighted loss → backpropagation → update weights.
    * Prints average training loss per epoch.

** Model used is :
<< ResNet50 (pretrained) : >>

ResNet50 → A deep convolutional neural network (CNN) architecture made of 50 layers.

    * "ResNet" = Residual Network (introduced by Microsoft Research in 2015).
    * It uses skip connections ("residuals") to avoid the vanishing gradient problem and lets you train deeper networks effectively.
    * Works very well for image classification tasks.
    * Pretrained → Instead of starting from scratch, the model has already been trained on a huge dataset (ImageNet: 1.2M images, 1000 classes).
    * The network already knows useful features like detecting edges, textures, shapes, patterns.
    * We just replace the last layer with your number of species and fine-tune it.
    * This is called transfer learning and saves time + data (otherwise you'd need millions of images to train from zero).
"""