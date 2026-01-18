from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import os
import re
import pytesseract

from app.inference import YOLODetector

app = FastAPI(title="YOLO ONNX Pothole Detection Service")

# Load YOLO once
detector = YOLODetector(
    model_path="model/yolov5n.onnx",
    conf_thres=0.6,
    iou_thres=0.5
)


# ---------------- VIDEO ANALYSIS ---------------- #

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    video_path = f"/tmp/{file.filename}"

    with open(video_path, "wb") as f:
        f.write(await file.read())

    results = process_video(video_path)

    return {
        "total_detections": len(results),
        "detections": results
    }


def process_video(video_path, every_n_seconds=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_interval = max(1, int(fps * every_n_seconds))
    frame_count = 0

    output = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # 1️⃣ YOLO
            detections, _ = detector.detect(frame)

            # 2️⃣ OCR
            gps_text = read_gps_text(frame)
            lat, lon = parse_lat_lon(gps_text)

            if detections and lat is not None and lon is not None:
                output.append({
                    "frame": frame_count,
                    "latitude": lat,
                    "longitude": lon,
                    "detections": detections
                })

        frame_count += 1

    cap.release()
    return output


# ---------------- OCR HELPERS ---------------- #

def read_gps_text(frame):
    h, w, _ = frame.shape

    # Crop bottom 20% of frame
    gps_crop = frame[int(h * 0.80):h, 0:w]

    text = pytesseract.image_to_string(
        gps_crop,
        config="--psm 6"
    )

    return text


def parse_lat_lon(text):
    lat = lon = None

    lat_match = re.search(r"Lat[:\s]*([0-9.+-]+)", text)
    lon_match = re.search(r"(Lon|Lng)[:\s]*([0-9.+-]+)", text)

    if lat_match:
        lat = float(lat_match.group(1))

    if lon_match:
        lon = float(lon_match.group(2))

    return lat, lon


# ---------------- IMAGE DETECTION (UNCHANGED) ---------------- #

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
