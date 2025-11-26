import json
from io import BytesIO
import os

import numpy as np
import streamlit as st
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input


@st.cache_resource
def load_model_and_labels(
    model_path="models/myna_model.keras",
    labels_path="models/labels.json",
    fallback_model="model.h5",
    fallback_labels="model_labels.json"
):
    # 優先使用 models/ 裡的
    if os.path.exists(model_path) and os.path.exists(labels_path):
        try:
            model = tf.keras.models.load_model(model_path)
            with open(labels_path, "r", encoding="utf-8") as f:
                labels = json.load(f)
            st.write(f"✅ Loaded model from {model_path}")
            return model, labels
        except Exception as e:
            st.warning(f"Failed to load model from {model_path}: {e}")

    # fallback
    if os.path.exists(fallback_model) and os.path.exists(fallback_labels):
        try:
            model = tf.keras.models.load_model(fallback_model)
            with open(fallback_labels, "r", encoding="utf-8") as f:
                labels = json.load(f)
            st.write(f"⚠️ Loaded fallback model from {fallback_model}")
            return model, labels
        except Exception as e:
            st.error(f"Failed to load fallback model: {e}")
            return None, None

    st.error("❌ No model found!")
    return None, None


def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((256, 256))
    arr = np.array(image).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def predict(model, labels, image: Image.Image):
    x = preprocess_image(image)
    preds = model.predict(x)[0]
    if labels is None:
        labels = [str(i) for i in range(len(preds))]
    items = list(zip(labels, preds.tolist()))
    items.sort(key=lambda t: t[1], reverse=True)
    return items


def main():
    st.title("八哥辨識 (Myna Classifier)")
    st.write("Upload a photo of a myna and the model will show predicted probabilities.")

    model, labels = load_model_and_labels()
    if model is None:
        st.warning("Model not found. Please create `model.h5` and `model_labels.json` or put a model in `models/` folder.")

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded is not None and model is not None:
        image = Image.open(BytesIO(uploaded.read()))
        st.image(image, caption="Uploaded", use_column_width=True)
        st.write("Classifying...")
        items = predict(model, labels, image)
        for label, prob in items:
            st.write(f"**{label}**: {prob:.4f}")


if __name__ == "__main__":
    main()
