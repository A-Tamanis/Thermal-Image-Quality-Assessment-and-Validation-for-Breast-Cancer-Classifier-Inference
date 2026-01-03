from ultralytics import YOLO

# Load YOLOv8n detection model (pretrained on COCO)
model = YOLO("runs_90_box/yolov8s_90_box/weights/last.pt")

# Train
model.train(
    data="breast_data_90.yaml",
    epochs=8,
    imgsz=640,
    batch=4,
    device="cpu",        # 0 = first GPU, or "cpu"
    project="runs_90_box",
    name="yolov8s_90_box",
    patience=10
)

# Optional: run validation on the best model
model.val(
    data="breast_data_90.yaml",
    imgsz=640,
    split="test"      # or "test"
)
