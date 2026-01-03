from pathlib import Path
from ultralytics import YOLO

# Path to your dataset root (where train/ val/ test/ live)
DATASET_ROOT = Path("data_sq")  # change if needed

def main():
    # 1. Load a pre-trained YOLOv8 classification model
    #    You can use: yolov8n-cls.pt, yolov8s-cls.pt, etc.
    model = YOLO("runs/classify/train6/weights/best.pt")

    # 2. Train the model
    # model.train(
    #     data=str(DATASET_ROOT),  # folder with train/val/test
    #     epochs=20,
    #     imgsz=640,               # common size for classification
    #     batch=2,                # adjust based on your RAM/VRAM
    #     device="cpu",            # use "0" for GPU if you have one
    # )

    # 3. Validate the model (by default on the val/ split)
    model.val(
        data=str(DATASET_ROOT),  # same dataset root
        split="test",             # or "test" to eval on test/
    )

if __name__ == "__main__":
    main()
