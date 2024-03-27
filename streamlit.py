import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import time

# Load the machine learning model from the directory
def load_model(model_dir):
    model = tf.keras.models.load_model(model_dir)
    return model

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to the required input shape
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    # Convert the image to numpy array
    img_array = np.array(image)
    # Normalize the pixel values
    img_array = img_array / 255.0
    # Expand the dimensions to match the model's input shape
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict plant disease directly in Streamlit
def predict(image, model):
    # Preprocess the image
    img_array = preprocess_image(image)
    # Perform prediction using the loaded model
    predictions = model.predict(img_array)
    # Convert the prediction to class label and confidence
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
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

    # Load the model from the directory
    model_dir = "./models/1/"
    model = load_model(model_dir)

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
