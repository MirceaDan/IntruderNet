# --- confusion matrix for model ---
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

# ======================
# CONFIG
# ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "model_3classes.pth"   # schimbă cu path-ul tău
EVAL_DIR = "evaluation"              # folderul cu imagini complet noi
BATCH_SIZE = 32
NUM_CLASSES = 3

CLASS_NAMES = ['misc', 'nothing', 'rabbit']  # MUST match training order

# ======================
# TRANSFORMS (NO AUGMENT)
# ======================
eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ======================
# DATASET & LOADER
# ======================
eval_dataset = datasets.ImageFolder(
    root=EVAL_DIR,
    transform=eval_transforms
)

eval_loader = DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0  # safe default (mai ales pe Raspi)
)

# ======================
# LOAD MODEL
# ======================
model = torch.load(MODEL_PATH, map_location=DEVICE)

# dacă ai salvat state_dict
if isinstance(model, dict):
    raise RuntimeError(
        "Ai încărcat un state_dict. Creează arhitectura modelului înainte!"
    )

model.to(DEVICE)
model.eval()

# ======================
# INFERENCE
# ======================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in eval_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ======================
# CONFUSION MATRIX
# ======================
cm = confusion_matrix(all_labels, all_preds)

print("\nConfusion Matrix:")
print(cm)

# ======================
# PER-CLASS ACCURACY
# ======================
print("\nAccuracy per class:")
for i, class_name in enumerate(CLASS_NAMES):
    class_mask = all_labels == i
    class_acc = accuracy_score(
        all_labels[class_mask],
        all_preds[class_mask]
    )
    print(f"{class_name:10s}: {class_acc:.4f}")

# ======================
# GLOBAL ACCURACY
# ======================
global_acc = accuracy_score(all_labels, all_preds)
print(f"\nGlobal Accuracy: {global_acc:.4f}")