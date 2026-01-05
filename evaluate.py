import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import defaultdict
import os

# =====================
# CONFIG
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
TEST_DIR = "test"
MODEL_PATH = "vit_intruder.pth"

CLASS_NAMES = ["misc", "nothing", "rabbit"]
NUM_CLASSES = len(CLASS_NAMES)

# =====================
# TRANSFORMS (NO AUG)
# =====================
test_transforms = transforms.Compose([
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
test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transforms
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# =====================
# LOAD MODEL
# =====================
model = torch.load(MODEL_PATH, map_location=DEVICE)
model.eval()
model.to(DEVICE)

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
    for images, labels in test_loader:
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
print("\n=== TEST RESULTS ===")
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
