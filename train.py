import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization
from PIL import ImageFile
from torchvision import transforms, datasets
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Configurare ---
DATA_DIR = "./dataset"
#dataset structure:
#dataset/
#    train/
#        rabbit/
#        misc/
#        nothing/
#    test/
#        rabbit/
#        misc/
#        nothing/

BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 10
NUM_CLASSES = 3 # for now rabbit, nothing future: 3rd class misc
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "./model/vit_intruder.pth"

# --- Augmentări și transformări ---
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- Dataset și DataLoader ---
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform = train_transforms)
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform = train_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# --- Dataset Weights ---
train_dir = os.path.join(DATA_DIR, "train")
classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]

# dictionar cu count pe clasa
class_counts = {}

for cls in classes:
    cls_path = os.path.join(train_dir, cls)
    n_files = len([f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))])
    class_counts[cls] = n_files

print(class_counts)

counts = torch.tensor([class_counts[cls] for cls in classes], dtype=torch.float)

weights = 1.0 / counts
class_weights = weights / weights.sum()

print(class_weights)

# --- Model ViT Tiny ---
weights = ViT_B_16_Weights.DEFAULT
model = vit_b_16(weights=weights)
in_features = model.heads.head.in_features
model.heads.head = nn.Linear(in_features, NUM_CLASSES)
model = model.to(DEVICE)

# --- Weighted Loss și Optimizer ---
criterion = torch.nn.CrossEntropyLoss(weight = class_weights.to(DEVICE))
optimizer = optim.AdamW(model.parameters(), lr=LR)

# --- Training loop ---
for epoch in range(EPOCHS) :
    start_time = time.time()
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss /total
    train_acc = correct / total

    #Test
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_loss /= test_total
    test_acc = test_correct / test_total
    
    epoch_time = time.time() - start_time

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}" | "
          f"{epoch_time/60:.2f} minutes")
    
# --- Quantizare pentru 8 biti ---
model.cpu()
model.eval()

model.qconfig = torch.quantization.get_default_qconfig('qnnpack') # fbgemm pentru x86, qnnpack pentru arm
torch.quantization.prepare(model, inplace=True)

# --- use some images from test for calibration
print("Running calibraiton for quantization...")
with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        model(images) # forward pass for calibration
        if i >= 10: # 10 batches enough for quantization
            break

# convert to 8 bit
torch.quantization.convert(model, inplace=True)
print("8-bit Quantization compelte")


# --- Salvare model ---
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model salvat la {MODEL_SAVE_PATH}")