"""
A script to load a ResNet50 model from a checkpoint and perform image classification.
Supports various checkpoint formats and reconstructs the final classifier layer accordingly.
The script also includes image preprocessing and outputs the top-k predictions.
"""
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import warnings

# ---------------- transforms ----------------
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---------------- loader ----------------
def load_model(checkpoint_path, device, default_num_classes=170):
    """
    Load a ResNet50 and reconstruct its final fc layer to match the checkpoint keys.
    Supports:
     - checkpoint saved as model.state_dict()
     - checkpoint saved as {"model_state": state_dict, ...}
    It inspects fc.* keys (e.g. fc.1.weight, fc.weight, fc.0.weight / fc.3.weight) and
    rebuilds the classifier accordingly.
    Returns (model, class_names)
    """
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # find the actual state_dict and class_names if present
    if isinstance(ck, dict) and "model_state" in ck:
        state = ck["model_state"]
    elif isinstance(ck, dict) and all(isinstance(v, torch.Tensor) for v in ck.values()):
        # appears to be a raw state_dict
        state = ck
    elif isinstance(ck, dict) and any(k.startswith("fc.") or k.startswith("layer") for k in ck.keys()):
        # raw state dict with tensor values
        state = ck
    else:
        # fallback: maybe checkpoint is a saved dict containing tensors under different key
        # try to find a nested dict that looks like a state dict
        state = None
        for v in ck.values() if isinstance(ck, dict) else []:
            if isinstance(v, dict) and any(k.startswith("fc.") for k in v.keys()):
                state = v
                break
        if state is None:
            raise RuntimeError("Could not locate model state_dict in checkpoint.")

    # attempt to read class_names from checkpoint (if present)
    class_names = None
    if isinstance(ck, dict):
        if "class_names" in ck:
            class_names = ck["class_names"]
        elif "classes" in ck:
            class_names = ck["classes"]
        elif "class_to_idx" in ck:
            # convert dict to list by sorting by value
            try:
                class_to_idx = ck["class_to_idx"]
                class_names = [None] * (max(class_to_idx.values()) + 1)
                for k, v in class_to_idx.items():
                    class_names[v] = k
            except Exception:
                class_names = None

    # Inspect fc keys to determine layout
    # prefer the most specific checks first
    if "fc.1.weight" in state:
        # typical: Sequential(Dropout, Linear) -> linear at index 1
        num_classes = state["fc.1.weight"].shape[0]
        in_feats = state["fc.1.weight"].shape[1]
        model = resnet50(weights=None)
        # rebuild exactly as training script
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_feats, num_classes)
        )
    elif "fc.weight" in state:
        # simple single Linear layer
        num_classes = state["fc.weight"].shape[0]
        in_feats = state["fc.weight"].shape[1]
        model = resnet50(weights=None)
        model.fc = nn.Linear(in_feats, num_classes)
    elif "fc.0.weight" in state and "fc.3.weight" in state:
        # Sequential(Linear, ReLU, Dropout, Linear) style
        hidden = state["fc.0.weight"].shape[0]
        in_feats = state["fc.0.weight"].shape[1]
        num_classes = state["fc.3.weight"].shape[0]
        model = resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Linear(in_feats, hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, num_classes)
        )
    else:
        # fallback: try to inspect any fc.*weight key to extract shapes
        fc_weight_keys = [k for k in state.keys() if k.startswith("fc") and k.endswith(".weight")]
        if len(fc_weight_keys) > 0:
            # pick the last fc weight (likely final linear)
            last_fc = sorted(fc_weight_keys)[-1]
            num_classes = state[last_fc].shape[0]
            in_feats = state[last_fc].shape[1]
            model = resnet50(weights=None)
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_feats, num_classes)
            )
            warnings.warn(f"Unknown fc layout; guessed final fc key '{last_fc}'.")
        else:
            raise RuntimeError("Could not detect classifier keys (no fc.*weight found).")

    model = model.to(device)

    # load state dict (handle both nested and raw)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        # try non-strict load (best-effort)
        warnings.warn(f"Strict load failed: {e}\nTrying load_state_dict(..., strict=False)")
        model.load_state_dict(state, strict=False)

    # finalize class_names
    if class_names is None:
        # best-effort: try to build placeholders
        try:
            n = num_classes
        except NameError:
            n = default_num_classes
        class_names = [f"class_{i}" for i in range(n)]

    return model, class_names

# ---------------- predict ----------------
def predict_image(model, image_path, class_names, device, topk=5):
    image = Image.open(image_path).convert("RGB")
    image_tensor = test_transforms(image).unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_probs, top_idxs = probs.topk(topk, dim=1)
    top_probs = top_probs[0].cpu().numpy()
    top_idxs = top_idxs[0].cpu().numpy()
    results = [(class_names[int(idx)], float(top_probs[i])) for i, idx in enumerate(top_idxs)]
    return results

# ---------------- main (user input) ----------------
if __name__ == "__main__":
    model_path = input("Enter model checkpoint path (.pth): ").strip()
    image_path = input("Enter image path: ").strip()

    if not os.path.exists(model_path):
        print("Model path does not exist:", model_path)
        raise SystemExit(1)
    if not os.path.exists(image_path):
        print("Image path does not exist:", image_path)
        raise SystemExit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(model_path, device)

    results = predict_image(model, image_path, class_names, device, topk=3)

    print("\nTop predictions:")
    for name, prob in results:
        print(f"{name}: {prob*100:.2f}%")
