import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, Conv2D
from PIL import Image
import os
from keras.saving import register_keras_serializable

# ---------------------------------------
# Custom Self-Attention Block
# ---------------------------------------
@register_keras_serializable()
class SelfAttentionBlock(Layer):
    def __init__(self, filters, **kwargs):
        super(SelfAttentionBlock, self).__init__(**kwargs)
        self.filters = filters
        self.query_conv = Conv2D(self.filters // 8, (1, 1), padding="same")
        self.key_conv = Conv2D(self.filters // 8, (1, 1), padding="same")
        self.value_conv = Conv2D(self.filters, (1, 1), padding="same")

    def call(self, x):
        q = self.query_conv(x)
        k = self.key_conv(x)
        v = self.value_conv(x)

        q_reshaped = tf.reshape(q, [tf.shape(x)[0], -1, self.filters // 8])
        k_reshaped = tf.reshape(k, [tf.shape(x)[0], -1, self.filters // 8])
        v_reshaped = tf.reshape(v, [tf.shape(x)[0], -1, self.filters])

        attention = tf.matmul(q_reshaped, k_reshaped, transpose_b=True)
        attention = tf.nn.softmax(attention, axis=-1)
        attention_output = tf.matmul(attention, v_reshaped)
        attention_output = tf.reshape(attention_output, tf.shape(x))

        return tf.add(attention_output, x)

    def get_config(self):
        config = super(SelfAttentionBlock, self).get_config()
        config.update({"filters": self.filters})
        return config

# ---------------------------------------
# Load All Models
# ---------------------------------------
print("🔄 Loading models...")
models = {}
try:
    models["U-Net"] = load_model("model/u_net_model.keras")
    models["Custom CNN"] = load_model("model/custom_cnn_model.keras")
    models["Self-Attention U-Net"] = load_model(
        "model/u_netself_attention-25.keras",
        custom_objects={"SelfAttentionBlock": SelfAttentionBlock}
    )
    print("✅ All models loaded successfully.")
except Exception as e:
    print("❌ Error loading models:", e)

# ---------------------------------------
# Preprocess and Postprocess Functions
# ---------------------------------------
def preprocess_image(image):
    try:
        image = image.resize((256, 256))
        img_array = np.array(image) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print("⚠️ Preprocessing error:", e)
        return None

def postprocess_mask(mask, binary=False):
    try:
        mask = np.squeeze(mask)
        if binary:
            mask = (mask > 0.5).astype(np.uint8) * 255
        else:
            mask = (mask * 255).astype(np.uint8)
        return Image.fromarray(mask)
    except Exception as e:
        print("⚠️ Postprocessing error:", e)
        return None

# ---------------------------------------
# Prediction Function
# ---------------------------------------
def segment_roads(image, binary_mask=False):
    print(f"🟡 Prediction started... Binary Mode: {binary_mask}")
    input_tensor = preprocess_image(image)
    if input_tensor is None:
        return [None] * 3

    outputs = []
    try:
        for name in ["U-Net", "Custom CNN", "Self-Attention U-Net"]:
            prediction = models[name].predict(input_tensor)
            mask = postprocess_mask(prediction, binary=binary_mask)
            outputs.append(mask)
        print("✅ Predictions complete.")
        return outputs
    except Exception as e:
        print("❌ Prediction error:", e)
        return [None] * 3

# ---------------------------------------
# Gradio UI with Professional Layout
# ---------------------------------------
with gr.Blocks(title="Road Segmentation Comparison") as demo:
    gr.Markdown(
        """
        # 🛣️ Road Segmentation Model Comparison  
        Upload a satellite image to see how different deep learning models detect roads.  
        You can also choose to see either **raw prediction masks** or **binary thresholded masks**.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Satellite Image")
            binary_checkbox = gr.Checkbox(label="Show Binary Mask Output", value=False)
            run_button = gr.Button("🔍 Run Prediction")
        with gr.Column(scale=2):
            unet_output = gr.Image(label="U-Net Prediction")
            cnn_output = gr.Image(label="Custom CNN Prediction")
            attention_output = gr.Image(label="Self-Attention U-Net Prediction")

    # Connect logic
    run_button.click(
        fn=segment_roads,
        inputs=[image_input, binary_checkbox],
        outputs=[unet_output, cnn_output, attention_output]
    )

    gr.Examples(
        examples=[["sample_inputs/test1.jpg", False]],
        inputs=[image_input, binary_checkbox]
    )

# ---------------------------------------
# Launch
# ---------------------------------------
demo.launch()
