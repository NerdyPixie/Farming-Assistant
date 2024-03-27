import streamlit as st
import requests
from PIL import Image
import io
import time

# Function to send image for prediction to FastAPI backend
def predict(image):
    try:
        # Prepare image for sending
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        # Send image to FastAPI endpoint
        files = {'file': img_byte_arr}
        response = requests.post('http://localhost:8000/predict', files=files)
        
        if response.status_code == 200:
            data = response.json()
            return data['class'], data['confidence']
        else:
            return None, None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Plant Disease Classification",
        page_icon="🌱",
        layout="wide"
    )

    st.title("Plant Disease Classification 🌱")
    st.markdown("---")

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
                predicted_class, confidence = predict(image)
                time.sleep(2)  # Simulate delay for demonstration purposes
                if predicted_class is not None:
                    st.markdown("---")
                    st.success(f"🌿 Predicted Class: **{predicted_class}**")
                    st.info(f"🎯 Confidence: {confidence:.2f}")
                else:
                    st.error("Failed to get prediction from server.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
