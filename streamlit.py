import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.models import load_model as keras_load_model
import time

# Load the machine learning model
def load_machine_learning_model():
    # Load your TensorFlow/Keras model here
    # Example:
    model = keras_load_model("models/1/1.keras")
    return model

# Function to preprocess the image
def preprocess_image(image):
    # Preprocess your image here (e.g., resize, normalize, etc.)
    # Example:
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to predict plant disease directly in Streamlit
def predict(image, model):
    # Preprocess the image
    img_array = preprocess_image(image)
    # Perform prediction using the loaded model
    predictions = model.predict(img_array)
    # Process predictions (optional, depends on your model output)
    # Example:
    predicted_class = "Placeholder Class"
    confidence = 0.75
    return predicted_class, confidence

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Plant Disease Classification",
        page_icon="🌱",
        layout="wide"
    )

    st.title("Plant Disease Classification 🌱")
    st.markdown("---")

    # Load the model
    model = load_machine_learning_model()

    # Upload image
    uploaded_file = st.file_uploader("Upload an image of a plant leaf", type=["jpg", "jpeg", "png"], help="Supported formats: JPG, JPEG, PNG")

    if uploaded_file is not None:
        st.markdown("---")

        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=300)

        # Predict button
        if st.button("Predict", key="predict_button"):
            with st.spinner('Predicting...'):
                predicted_class, confidence = predict(image, model)
                time.sleep(2)  # Simulate delay for demonstration purposes
                if predicted_class is not None:
                    st.markdown("---")
                    st.success(f"🌿 Predicted Class: **{predicted_class}**")
                    st.info(f"🎯 Confidence: {confidence:.2f}")
                else:
                    st.error("Failed to get prediction.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
