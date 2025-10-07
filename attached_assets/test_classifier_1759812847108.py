import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---------------- CONFIG ----------------
model_path = "/Users/mac/Desktop/my_model/Model_stuff/Models/mammal_detector.pth"
image_path = "/Users/mac/Desktop/test_imgs/images.jpeg"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 224
# ----------------------------------------

# Preprocessing (must match validation transform)
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def create_model():
    model = models.efficientnet_b0(weights=None)  # no pretrained weights in inference
    model.classifier = nn.Sequential(
        nn.Linear(1280, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 1),
    )
    return model

def load_model(model_path):
    model = create_model()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def predict(model, image_path, threshold=0.8):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(image).squeeze()
        prob = torch.sigmoid(logit).item()

    label = "Mammal" if prob > threshold else "Not a mammal"
    print(f"Prediction: {label}")

if __name__ == "__main__":
    model = load_model(model_path)
    predict(model, image_path)
