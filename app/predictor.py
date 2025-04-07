from app.model_loader import get_model

def predict_adult_content(img_tensor):
    model = get_model()
    prediction = model.predict(img_tensor)
    return float(prediction[0][0])
