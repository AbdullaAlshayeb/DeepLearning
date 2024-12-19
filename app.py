from dotenv import load_dotenv
load_dotenv()

from utils import *
import streamlit as st
from streamlit_drawable_canvas import st_canvas

@st.cache_resource
def load_weights():
    return load_dataset()

@st.cache_resource
def setup_models():
    return load_models()

load_weights()

resnet_model, densenet_model, xception_model = setup_models()

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
    model = resnet_model if model_choice == "ResNet" else densenet_model if model_choice == "DenseNet" else xception_model

    # Predict button
    if st.button("Predict"):
        # Make prediction
        try:
            image = preprocessing_image(image, model_type=model_choice.lower())
            prediction = predict_image(model, image)
            st.write(f"Predicted Class: {class_names[prediction]}")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
