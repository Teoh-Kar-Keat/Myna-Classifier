import json
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input


@st.cache_resource
def load_model_and_labels(model_path="model.h5", labels_path="model_labels.json"):
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None
    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
    except Exception:
        labels = None
    return model, labels


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
    # build a sorted list of (label, prob)
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
        st.warning("Model not found. Run `training.py` to create `model.h5` and `model_labels.json`, then refresh.")

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
