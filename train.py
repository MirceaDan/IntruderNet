import torch
import torch.nn as nn
import torch.optim as optim
import toch.quantization
from torchvision import transforms, datasets
from torchvision.models import vit_b_16
from torch.utils.data import DataLoader
import os

# --- Configurare ---
DATA_DIR = "./dataset"
#dataset structure:
#dataset/
#    train/
#        rabbit/
#        misc/
#        nothing/
#    val/
#        rabbit/
#        misc/
#        nothing/

BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 10
NUM_CLASSES = 3 # rabbit, misc, nothing
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "./vit_intruder.pth"

# --- Augmentări și transformări ---
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# --- Dataset și DataLoader ---
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform = train_transforms)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform = train_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# --- Model ViT Tiny ---
model = vit_b_16(pretrained=True)
model.heads = nn.Linear(model.heads.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# --- Loss și Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

# --- Training loop ---
for epoch in range(EPOCHS) :
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

    #Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= val_total
    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
# --- Quantizare pentru 8 biti ---
model.cpu()
model.eval()

model.qconfig = torch.quantization.get_default_qconfig('qnnpack') # fbgemm pentru x86, qnnpack pentru arm
torch.quantization.prepare(model, inplace=True)

# --- use some images from validation for calibration
print("Running calibraiton for quantization...")
with torch.no_grade():
    for i, (images, labels) in enumerate(val_loader):
        model(images) # forward pass for calibration
        if i >= 10: # 10 batches enough for quantization
            break

# convert to 8 bit
toch.quantization.convert(model, inplace=True)
print("8-bit Quantization compelte")


# --- Salvare model ---
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model salvat la {MODEL_SAVE_PATH}")