from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

app = FastAPI()

# Allow React Native to access your server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific IP for security
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your YOLOv8 model
model = YOLO("runs/detect/yolov8_cpu2/weights/best.pt")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    np_image = np.array(image)

    results = model(np_image, imgsz=640)[0]
    detections = results.boxes.data.cpu().numpy()

    response = []
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        label = model.names[int(cls_id)]
        response.append({
            "label": label,
            "confidence": float(conf),
            "box": [float(x1), float(y1), float(x2), float(y2)]
        })

    return {"detections": response}
