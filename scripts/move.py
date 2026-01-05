import os
import shutil
import random

# ---------------- CONFIG ----------------
DATASET_ROOT = "dataset"
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "validation")

CLASSES = ["rabbit", "nothing", "misc"]
SPLIT_RATIO = 0.10
SEED = 42
# ----------------------------------------

random.seed(SEED)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def move_split(class_name):
    train_class_dir = os.path.join(TRAIN_DIR, class_name)
    val_class_dir = os.path.join(VAL_DIR, class_name)

    ensure_dir(val_class_dir)

    files = [
        f for f in os.listdir(train_class_dir)
        if os.path.isfile(os.path.join(train_class_dir, f))
    ]

    n_total = len(files)
    n_move = int(n_total * SPLIT_RATIO)

    selected = random.sample(files, n_move)

    for fname in selected:
        src = os.path.join(train_class_dir, fname)
        dst = os.path.join(val_class_dir, fname)

        shutil.move(src, dst)

    print(f"[{class_name}] moved {n_move}/{n_total} files to validation")


def main():
    for cls in CLASSES:
        move_split(cls)


if __name__ == "__main__":
    main()
