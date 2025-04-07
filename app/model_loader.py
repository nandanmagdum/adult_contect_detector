import tensorflow_hub as hub
from tensorflow.keras.models import load_model

def get_model():
    model = load_model("model.h5", custom_objects={"KerasLayer": hub.KerasLayer})
    return model
