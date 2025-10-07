import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def create_mammal_detector_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(1280, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 1),
    )
    return model

def create_f1_model(num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class MammalDetector:
    def __init__(self, model_path='models/mammal_detector.pth'):
        self.model = create_mammal_detector_model()
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        self.model.to(device)
        self.model.eval()
    
    def predict(self, image_path, threshold=0.8):
        image = Image.open(image_path).convert("RGB")
        image_tensor = val_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logit = self.model(image_tensor).squeeze()
            prob = torch.sigmoid(logit).item()
        
        return {
            'is_mammal': prob > threshold,
            'confidence': prob
        }

class SpeciesClassifier:
    def __init__(self, model_path='models/best_f1_model.pth', classes_path='models/species_classes.json'):
        with open(classes_path, 'r') as f:
            self.class_names = json.load(f)
        
        self.num_classes = len(self.class_names)
        self.model = create_f1_model(self.num_classes)
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
    
    def predict(self, image_path, confidence_threshold=0.6):
        image = Image.open(image_path).convert("RGB")
        image_tensor = val_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            top_prob, top_idx = probs.topk(1, dim=1)
        
        confidence = top_prob[0].item()
        predicted_class = self.class_names[top_idx[0].item()]
        
        return {
            'species': predicted_class.replace('_', ' '),
            'confidence': confidence,
            'is_confident': confidence > confidence_threshold
        }
