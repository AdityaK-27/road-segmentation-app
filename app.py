import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from self_attention import SelfAttentionBlock  # Import the registered layer
import os

# ------------------------------------------
# Page Config & UI Setup
# ------------------------------------------
st.set_page_config(
    page_title="Road Segmentation Model Comparison",
    layout="wide",
    page_icon="🛣️"
)

st.title("🛣️ Road Segmentation Model Comparison")
st.markdown("""
Upload a satellite image and compare how three different deep learning models (U-Net, Custom CNN, and Self-Attention U-Net) perform road segmentation.

Use the checkbox below to choose between raw prediction or binary (thresholded) output.
""")

# ------------------------------------------
# Model Loader
# ------------------------------------------
@st.cache_resource
def load_models():
    models = {}
    try:
        models["U-Net"] = tf.keras.models.load_model("models/u_net_model.keras")
        models["Custom CNN"] = tf.keras.models.load_model("models/custom_cnn_model.keras")
        models["Self-Attention U-Net"] = tf.keras.models.load_model(
            "models/u_netself_attention-25.keras",
            custom_objects={"SelfAttentionBlock": SelfAttentionBlock}
        )
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
    return models

models = load_models()

# ------------------------------------------
# Preprocess & Postprocess
# ------------------------------------------
def preprocess_image(image: Image.Image):
    image = image.resize((256, 256))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

def postprocess_mask(mask, binary=False):
    mask = np.squeeze(mask)
    if binary:
        mask = (mask > 0.5).astype(np.uint8) * 255
    else:
        mask = (mask * 255).astype(np.uint8)
    return Image.fromarray(mask)

# ------------------------------------------
# Sidebar Controls
# ------------------------------------------
st.sidebar.header("⚙️ Options")
binary_mask = st.sidebar.checkbox("Show Binary Mask (Threshold > 0.5)", value=False)
example_path = os.path.join("assets", "test1.jpg")
if os.path.exists(example_path):
    if st.sidebar.button("📂 Load Example Image"):
        uploaded_file = example_path
    else:
        uploaded_file = st.file_uploader("Upload Satellite Image", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.file_uploader("Upload Satellite Image", type=["jpg", "jpeg", "png"])

# ------------------------------------------
# Main Logic
# ------------------------------------------
if uploaded_file:
    if isinstance(uploaded_file, str):
        image = Image.open(uploaded_file)
    else:
        image = Image.open(uploaded_file).convert("RGB")

    st.subheader("🖼️ Uploaded Image")
    st.image(image, use_column_width=True)

    st.subheader("🔍 Model Predictions")
    input_tensor = preprocess_image(image)

    cols = st.columns(3)
    model_names = ["U-Net", "Custom CNN", "Self-Attention U-Net"]

    for idx, name in enumerate(model_names):
        try:
            prediction = models[name].predict(input_tensor)
            mask = postprocess_mask(prediction, binary=binary_mask)
            cols[idx].image(mask, caption=f"{name} Output", use_column_width=True)
        except Exception as e:
            cols[idx].error(f"❌ {name} Error: {e}")
else:
    st.info("👈 Please upload an image to get started.")

# ------------------------------------------
# Footer
# ------------------------------------------
st.markdown("""
---
© 2025 [Your Name] — Road Segmentation App built with TensorFlow & Streamlit
""")
