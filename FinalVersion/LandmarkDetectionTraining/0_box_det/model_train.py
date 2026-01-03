from ultralytics import YOLO

# Load YOLOv8n detection model (pretrained on COCO)
model = YOLO("yolov8s.pt")

# Train
model.train(
    data="breast_data_0.yaml",
    epochs=20,
    imgsz=640,
    batch=2,
    device="cpu",        # 0 = first GPU, or "cpu"
    project="runs_0_box",
    #name="yolov8s_0_box",
    patience=10
)

# Optional: run validation on the best model
model.val(
    data="breast_data_0.yaml",
    imgsz=640,
    split="test"      # or "test"
)
