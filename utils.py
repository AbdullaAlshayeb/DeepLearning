import numpy as np
import kaggle
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.saving import register_keras_serializable
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

@register_keras_serializable()
def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.round(tf.nn.sigmoid(y_pred))  # Round predicted values to 0 or 1

    tp = tf.reduce_sum(y_true * y_pred)  # True positives
    fp = tf.reduce_sum((1 - y_true) * y_pred)  # False positives
    fn = tf.reduce_sum(y_true * (1 - y_pred))  # False negatives

    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return f1

def load_dataset():
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('oyounis/model-weights-dataset', path='data/', unzip=True)

def load_models():
    resnet_model = load_model("data/resnet.keras", custom_objects={'f1_score': f1_score})
    densenet_model = load_model("data/densenet.keras", custom_objects={'f1_score': f1_score})
    xception_model = load_model("data/xception.keras", custom_objects={'f1_score': f1_score})

    return resnet_model, densenet_model, xception_model

# Preprocessing Function
def preprocessing_image(img, model_type='resnet'):
    img = tf.image.rgb_to_grayscale(img) if model_type == 'resnet' else img
    img = tf.image.resize(img,(28, 28) if model_type == 'resnet' else (128, 128))
    img= tf.cast(img, tf.float32) / 255.0

    if len(img.shape) == 3:
        img = tf.expand_dims(img, axis=0)

    return img

# Predict output of image
def predict_image(model, image):
    pred = model.predict(image)
    pred = np.argmax(pred, axis=1)
    return pred[0]