from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware  # 👉 ADD THIS
from app.utils import preprocess_image
from app.predictor import predict_adult_content

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # allow_origins=["http://localhost:5173"],  # or ["*"] to allow all origins (not recommended for prod)
    allow_origins=["*"],  # or ["*"] to allow all origins (not recommended for prod)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_tensor = preprocess_image(image_bytes)
    score = predict_adult_content(img_tensor)
    return {
        "adult_content": score > 0.5,
        "confidence": round(score, 4)
    }
