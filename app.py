import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Set page configuration
st.set_page_config(page_title="Road Segmentation App", layout="wide")

# Title and description
st.title("🛣️ Road Segmentation Model Comparison")
st.markdown("Compare different road segmentation models (U-Net, Custom CNN, Self-Attention U-Net) on satellite images.")

# Load models
@st.cache_resource
def load_models():
    u_net_model = tf.keras.models.load_model("model/u_net_model.keras", compile=False)
    custom_cnn_model = tf.keras.models.load_model("model/custom_cnn_model.keras", compile=False)
    self_attention_model = tf.keras.models.load_model("model/u_netself_attention-25.keras", compile=False)
    return u_net_model, custom_cnn_model, self_attention_model

u_net_model, custom_cnn_model, self_attention_model = load_models()

# Preprocessing function
def preprocess_image(image):
    image = image.resize((256, 256))
    image_array = np.array(image).astype(np.float32) / 255.0
    if image_array.shape[-1] == 4:  # Remove alpha channel if exists
        image_array = image_array[..., :3]
    return np.expand_dims(image_array, axis=0)

# Prediction function
def predict_mask(model, image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0]
    prediction = (prediction > 0.5).astype(np.uint8) * 255
    return Image.fromarray(prediction.squeeze()).convert("L")

# Upload section
uploaded_image = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"])

# Image and predictions
if uploaded_image:
    image = Image.open(uploaded_image)

    # Columns for displaying each model output
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("U-Net Prediction")
        unet_mask = predict_mask(u_net_model, image)
        st.image(unet_mask, use_column_width=True)

    with col2:
        st.subheader("Custom CNN Prediction")
        cnn_mask = predict_mask(custom_cnn_model, image)
        st.image(cnn_mask, use_column_width=True)

    with col3:
        st.subheader("Self-Attention U-Net Prediction")
        attn_mask = predict_mask(self_attention_model, image)
        st.image(attn_mask, use_column_width=True)

    st.success("✅ Segmentation completed successfully!")
