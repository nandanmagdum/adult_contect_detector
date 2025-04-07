from tensorflow.keras.models import load_model

_model = load_model("model.h5")

def get_model():
    return _model
