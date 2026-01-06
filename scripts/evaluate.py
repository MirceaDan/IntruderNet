# --- model evaluation script ---
# --- vit_intruder.pth -> x86 ---
# --- vit_intruder_q.pth -> arm ---
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16
from collections import defaultdict
import os

# =====================
# CONFIG
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
VALIDATION_DIR = "./dataset/validation"
MODEL_PATH = "./model/vit_intruder.pth"

CLASS_NAMES = ["misc", "nothing", "rabbit"]
NUM_CLASSES = len(CLASS_NAMES)

# =====================
# TRANSFORMS (NO AUG)
# =====================
validation_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================
# DATASET
# =====================
validation_dataset = datasets.ImageFolder(
    VALIDATION_DIR,
    transform=validation_transforms
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# =====================
# LOAD MODEL
# =====================
model = vit_b_16(weights=None)

in_features = model.heads.head.in_features
model.heads.head = nn.Linear(in_features, NUM_CLASSES)

model.eval()

model.qconfig = torch.quantization.get_default_qconfig("qnnpack")
torch.quantization.prepare(model, inplace=True)
torch.quantization.convert(model, inplace=True)

state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict)

criterion = nn.CrossEntropyLoss()

# =====================
# METRICS
# =====================
correct_per_class = defaultdict(int)
total_per_class = defaultdict(int)
total_correct = 0
total_samples = 0
total_loss = 0.0

# =====================
# INFERENCE LOOP
# =====================
with torch.no_grad():
    for images, labels in validation_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)

        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        for label, pred in zip(labels, preds):
            total_per_class[label.item()] += 1
            if label == pred:
                correct_per_class[label.item()] += 1

# =====================
# RESULTS
# =====================
print("\n=== VALIDATION RESULTS ===")
print(f"Overall Accuracy: {total_correct / total_samples:.4f}")
print(f"Average Loss: {total_loss / total_samples:.4f}\n")

print("Accuracy per class:")
for idx, class_name in enumerate(CLASS_NAMES):
    if total_per_class[idx] == 0:
        acc = 0.0
    else:
        acc = correct_per_class[idx] / total_per_class[idx]

    print(f"  {class_name:10s}: {acc:.4f} "
          f"({correct_per_class[idx]}/{total_per_class[idx]})")
