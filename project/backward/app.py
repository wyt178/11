# A Demo 
# 提供 predict接口
# 接受图片
# 调用模型接口
# 返回图片

from fastapi import FastAPI,File,UploadFile 
from fastapi.responses import JSONResponse, FileResponse 
from ultralytics import YOLO
from PIL import Image 
import io,os,uuid 

app = FastAPI()
model = YOLO("runs/detect/train/weights/best.pt")

@app.get("/")
def read_root():
    return {"message": "yolo image recognition API running."}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))

    results = model(img)

    save_dir = "static/results"
    os.makedirs(save_dir,exist_ok=True)
    result_path = f"{save_dir}/{uuid.uuid4().hex}.jpg"
    results[0].save(filename=result_path)

    detections = []
    for box in results[0].boxes:
        detections.append({
            "label": model.names[int(box.cls)],
            "confidence": float(box.conf),
            "bbox": box.xyxy[0].tolist(),
        })

    return JSONResponse({
        "detections": detections,
        "result_img": result_path,
    })