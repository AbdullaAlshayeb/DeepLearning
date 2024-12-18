import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

class_names = []  # Add class names mr. omar abdelnasser

# Preprocessing Function for ResNet
def preprocessing_image_res(img, target_size=(28, 28)):
    img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize(img,target_size)
    img= tf.cast(img, tf.float32) / 255. 

    return img

# Preprocessing Function for other models
def preprocessing_image(img, target_size=(224, 224)):
    img = tf.image.resize(img,target_size)
    img= tf.cast(img, tf.float32) / 255. 

    return img

# Predict output of image
def predict_image(model, image):
    pred = model.predict(image)
    pred = np.argmax(pred, axis=1)

    return pred[0]


# Please add the fucken models
def upload_model(model_name, model_dir="models/"):
    model_path = f"{model_dir}/{model_name}.h5"
    try:
        model = load_model(model_path)
        return model
    except:
        st.error(f"Model '{model_name}' not found in {model_dir}.")
        return None


def main():
    st.title("Letter Recognition App")
    st.write("Draw a character, upload an image, and predict the class.")
    
    # Option to draw or upload image
    mode = st.radio("Choose Mode", ("Draw", "Upload Image"))
    
    if mode == "Draw":
        canvas_state = st_canvas(
            stroke_width=15,  # Pen width
            stroke_color="white",  # Pen color
            background_color="black",  # Canvas background
            width=300,  # Canvas width
            height=300,  # Canvas height
            drawing_mode="freedraw",  # Drawing mode
        )

        # Get the drawing from the canvas
        if canvas_state.image_data is not None:
            # Convert the canvas image (RGBA) to RGB for model input
            image = canvas_state.image_data.astype(np.float32)[:, :, :3]

    
    elif mode == "Upload Image":
        # File uploader for the image
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "gif"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = tf.keras.utils.load_img(uploaded_file)
            st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Model selection dropdown
    model_choice = st.selectbox("Select Model", ["ResNet", "DenseNet", "Xception"])

    # Load the selected model
    model = upload_model(model_choice)


    if model:
        # Predict button
        if st.button("Predict"):
            # Make prediction
            try:
                if model_choice == "ResNet":
                    image = preprocessing_image_res(image)
                else:
                    image = preprocessing_image(image)
                prediction = predict_image(model, image)
                st.write(f"Predicted Class: {class_names[prediction]}")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
