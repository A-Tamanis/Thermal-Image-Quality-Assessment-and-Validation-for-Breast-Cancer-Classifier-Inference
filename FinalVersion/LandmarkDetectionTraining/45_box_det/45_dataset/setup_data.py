import os
import random
import shutil
from pathlib import Path

# --------- CONFIG ---------
PROJECT_ROOT = Path(__file__).resolve().parent

IMAGES_RAW  = PROJECT_ROOT / "1perPatient45right"   # folder with all images
LABELS_RAW  = PROJECT_ROOT / "labels_pose_45"   # folder with all txt labels

DATASET_ROOT = PROJECT_ROOT / "45metrics_dataset"

TRAIN_SPLIT = 0.8
VAL_SPLIT   = 0.1   # rest goes to test

IMG_EXTS = (".jpg", ".jpeg", ".png")  # adjust if needed
# --------------------------

random.seed(42)

def main():
    # Collect all image files
    all_images = [
        p for p in IMAGES_RAW.iterdir()
        if p.suffix.lower() in IMG_EXTS
    ]
    print(f"Found {len(all_images)} images")

    random.shuffle(all_images)

    n_total = len(all_images)
    n_train = int(n_total * TRAIN_SPLIT)
    n_val   = int(n_total * VAL_SPLIT)
    n_test  = n_total - n_train - n_val

    train_imgs = all_images[:n_train]
    val_imgs   = all_images[n_train:n_train + n_val]
    test_imgs  = all_images[n_train + n_val:]

    splits = {
        "train": train_imgs,
        "val":   val_imgs,
        "test":  test_imgs,
    }

    # Create folders
    for split in splits.keys():
        (DATASET_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Copy files
    for split, img_list in splits.items():
        print(f"\nProcessing {split} ({len(img_list)} images)")
        for img_path in img_list:
            stem = img_path.stem
            label_path = LABELS_RAW / f"{stem}.txt"

            if not label_path.exists():
                print(f"⚠️ No label for {img_path.name}, skipping")
                continue

            # Destination paths
            dst_img   = DATASET_ROOT / "images" / split / img_path.name
            dst_label = DATASET_ROOT / "labels" / split / label_path.name

            shutil.copy2(img_path, dst_img)
            shutil.copy2(label_path, dst_label)

    print("\nDataset created at:", DATASET_ROOT)

if __name__ == "__main__":
    main()
