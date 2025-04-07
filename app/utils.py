from PIL import Image
import numpy as np
import io

def preprocess_image(image_data, size=(224, 224)):
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize(size)
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)
