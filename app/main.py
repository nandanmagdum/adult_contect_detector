from fastapi import FastAPI, File, UploadFile
from app.utils import preprocess_image
from app.predictor import predict_adult_content

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_tensor = preprocess_image(image_bytes)
    score = predict_adult_content(img_tensor)
    return {
        "adult_content": score > 0.5,
        "confidence": round(score, 4)
    }
