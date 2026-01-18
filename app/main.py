from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from app.inference import YOLODetector

app = FastAPI(title="YOLO ONNX Pothole Detection Service")

# Load model once at startup
detector = YOLODetector(
    model_path="model/yolov5n.onnx",
    conf_thres=0.6,
    iou_thres=0.5
)

@app.post("/detect")
async def detect_pothole(file: UploadFile = File(...)):
    contents = await file.read()

    np_img = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Invalid image"}

    detections, infer_time = detector.detect(image)

    return {
        "detections": detections,
        "inference_time_ms": round(infer_time, 2)
    }
