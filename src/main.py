import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import os
import requests
from io import BytesIO

# Set page config
st.set_page_config(page_title="Damage Classifier", page_icon="🔍", layout="wide")

# Function to load the model (with caching)
@st.cache_resource
def load_model():
    # Check if model file exists, if not download it
    model_path = '../models/cnn.keras'
    
    if not os.path.exists(model_path):
        st.warning("Model file not found. Please upload a trained model.")
        return None
    
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model = load_model()

# Class names (adjust according to your dataset)
class_names = [
    "Level 0: No Damage",
    "Level 1: Minor Damage",
    "Level 2: Moderate Damage",
    "Level 3: Significant Damage",
    "Level 4: Severe Damage",
    "Level 5: Critical Damage"
]

# Preprocess the image
def preprocess_image(image):
    try:
        # Resize to match model's expected sizing
        image = image.resize((224, 224))
        # Convert to numpy array
        image_array = np.array(image)
        # Normalize pixel values to [0, 1]
        image_array = image_array / 255.0
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# Make prediction
def predict_damage(image_array):
    try:
        if model is None:
            return None
        
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        return predicted_class, confidence, predictions[0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None, None

# Main app
def main():
    st.title("Damage Level Classifier")
    st.write("Upload an image to classify the damage level (0-5)")

    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        """
        This app uses a pre-trained CNN model to classify damage into six levels:
        - Level 0: No Damage
        - Level 1: Minor Damage
        - Level 2: Moderate Damage
        - Level 3: Significant Damage
        - Level 4: Severe Damage
        - Level 5: Critical Damage
        
        Model trained on the Ripik Hackfest dataset from Kaggle.
        """
    )
    
    st.sidebar.header("Instructions")
    st.sidebar.write("1. Upload an image or provide a URL")
    st.sidebar.write("2. The model will predict the damage level")
    st.sidebar.write("3. View results and confidence levels")

    # Image upload options
    option = st.radio("Select input method:", ("Upload Image", "Image URL"))

    image = None
    
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
    else:
        url = st.text_input("Enter the image URL:")
        if url:
            try:
                response = requests.get(url)
                image = Image.open(BytesIO(response.content))
            except:
                st.error("Error loading image from URL")

    if image is not None:
        # Display the image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess and predict
        image_array = preprocess_image(image)
        
        if image_array is not None:
            with st.spinner("Analyzing damage..."):
                predicted_class, confidence, predictions = predict_damage(image_array)
            
            if predicted_class is not None:
                st.success(f"Predicted Damage Level: {class_names[predicted_class]}")
                st.success(f"Confidence: {confidence*100:.2f}%")
                
                # Display confidence bars for all classes
                st.subheader("Prediction Confidence for All Levels:")
                for i, (class_name, prob) in enumerate(zip(class_names, predictions)):
                    st.write(f"{class_name}:")
                    st.progress(float(prob))
                    st.write(f"{prob*100:.2f}%")
                    
                # Display interpretation
                st.subheader("Interpretation:")
                if predicted_class == 0:
                    st.info("No damage detected. The item is in perfect condition.")
                elif predicted_class == 1:
                    st.info("Minor damage detected. The item has slight imperfections but is still fully functional.")
                elif predicted_class == 2:
                    st.warning("Moderate damage detected. The item shows noticeable signs of wear but remains operational.")
                elif predicted_class == 3:
                    st.warning("Significant damage detected. The item has substantial damage that may affect functionality.")
                elif predicted_class == 4:
                    st.error("Severe damage detected. The item is heavily damaged and may not function properly.")
                else:
                    st.error("Critical damage detected. The item is severely compromised and likely non-functional.")

if __name__ == "__main__":
    main()