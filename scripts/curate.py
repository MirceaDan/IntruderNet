# --- check and remove truncated images from entire dataset (train, test, validation) --- 
import os
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False  # vrem să CRAPE ca să le detectăm

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

ROOT_DIR = "dataset"   # schimba dacă e nevoie


def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)


def clean_folder(root_dir):
    removed = 0
    checked = 0

    for root, _, files in os.walk(root_dir):
        for fname in files:
            if not is_image_file(fname):
                continue

            path = os.path.join(root, fname)
            checked += 1

            try:
                with Image.open(path) as img:
                    img.verify()  # verifică integritatea fișierului
            except Exception as e:
                print(f"[DELETE] {path} -> {e}")
                try:
                    os.remove(path)
                    removed += 1
                except Exception as rm_err:
                    print(f"[ERROR] could not delete {path}: {rm_err}")

    print("\n======================")
    print(f"Checked images : {checked}")
    print(f"Deleted images : {removed}")
    print("======================")


if __name__ == "__main__":
    clean_folder(ROOT_DIR)