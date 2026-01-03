from ultralytics import YOLO
from pose_eval_45 import evaluate_pose  
from draw_pose_45 import draw_pose_overlay
import cv2

# 1. Load model and image
model = YOLO("best.pt")
img = cv2.imread("p013.jpg")

# 2. Run detection
results = model(img)[0]
img_h, img_w = results.orig_shape
detections = results.boxes.data.cpu().numpy().tolist()  # [x1,y1,x2,y2,conf,cls]

# 3. Evaluate pose
pose_info = evaluate_pose(detections, img_w, img_h)

# 4. Draw overlay
annotated = draw_pose_overlay(img, pose_info, show_reasons=True, max_reasons=5)

# 5. Show or save
cv2.imshow("pose check", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
# or:
# cv2.imwrite("p191_pose_overlay.jpg", annotated)