import io
import torch
from torchvision import models, transforms
from PIL import Image

# Your confirmed class order
CLASS_NAMES = [
    "Amrapali",
    "Banana",
    "Bandigori",
    "Brunei King",
    "Harivanga",
    "Himsagar",
    "Kacha Mitha",
    "Surjapuri"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing exactly as you said: 224x224
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def load_model(weights_path="models/DenseNet121-MangoLeaf_best.pth"):
    model = models.densenet121(weights=None)

    num_ftrs = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_ftrs, 8)

    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model

def predict_image_bytes(model, image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = PREPROCESS(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_idx = torch.max(probs, dim=1)

    predicted_class = CLASS_NAMES[top_idx.item()]
    confidence = float(top_prob.item())

    return predicted_class, confidence
