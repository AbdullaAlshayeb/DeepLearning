import numpy as np
import kaggle
import tensorflow as tf
from tensorflow.keras.models import load_model

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

def load_dataset():
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('oyounis/model-weights-dataset', path='data/', unzip=True)

def load_models():
    resnet_model = load_model("data/resnet.keras")
    densenet_model = load_model("data/densenet.keras")
    xception_model = load_model("data/xception.keras")

    return resnet_model, densenet_model, xception_model

# Preprocessing Function
def preprocessing_image(img, model_type='resnet'):
    img = tf.image.rgb_to_grayscale(img) if model_type == 'resnet' else img
    img = tf.image.resize(img,(28, 28, 1) if model_type == 'resnet' else (224, 224, 3))
    img= tf.cast(img, tf.float32) / 255.0

    return img

# Predict output of image
def predict_image(model, image):
    pred = model.predict(image)
    pred = np.argmax(pred, axis=1)
    return pred[0]