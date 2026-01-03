from ultralytics import YOLO

# Load YOLOv8n detection model (pretrained on COCO)
model = YOLO("yolov8s.pt")

# Train
model.train(
    data="breast_data_45.yaml",
    epochs=30,
    imgsz=640,
    batch=4,
    device="cpu",        # 0 = first GPU, or "cpu"
    project="runs_45_box",
    name="yolov8s_45_box",
    patience=10
)

# Optional: run validation on the best model
model.val(
    data="breast_data.yaml",
    imgsz=640,
    split="val"      # or "test"
)
