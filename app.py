import json
from io import BytesIO
import os
import logging

import numpy as np
import streamlit as st
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input


# quiet TF logging a bit
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
logging.getLogger("tensorflow").setLevel(logging.ERROR)


@st.cache_resource
def load_model_and_labels(
    model_path="models/myna_model.keras",
    labels_path="models/labels.json",
    fallback_model="model.h5",
    fallback_labels="model_labels.json",
):
    """
    Tries to load a saved Keras model and a JSON labels file. Prefer models/ paths,
    then fallback files. Returns (model, labels, input_size) where input_size is
    a (height, width) tuple to be used for preprocessing.
    """
    def _load(m_path, l_path):
        model = tf.keras.models.load_model(m_path, compile=False)
        with open(l_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        # determine expected input size from model if possible
        try:
            shape = model.input_shape  # e.g. (None, H, W, C)
            if shape is None:
                input_size = (256, 256)
            else:
                # take indices 1 and 2 if available and not None
                h = shape[1] if len(shape) > 1 and shape[1] is not None else 256
                w = shape[2] if len(shape) > 2 and shape[2] is not None else 256
                input_size = (int(h), int(w))
        except Exception:
            input_size = (256, 256)
        return model, labels, input_size

    # prefer files under models/
    if os.path.exists(model_path) and os.path.exists(labels_path):
        try:
            model, labels, input_size = _load(model_path, labels_path)
            st.success(f"✅ Loaded model from {model_path}")
            return model, labels, input_size
        except Exception as e:
            st.warning(f"Failed to load model from {model_path}: {e}")

    # fallback
    if os.path.exists(fallback_model) and os.path.exists(fallback_labels):
        try:
            model, labels, input_size = _load(fallback_model, fallback_labels)
            st.warning(f"⚠️ Loaded fallback model from {fallback_model}")
            return model, labels, input_size
        except Exception as e:
            st.error(f"Failed to load fallback model: {e}")
            return None, None, None

    st.error("❌ No model found! Place a model in models/ or create model.h5 + model_labels.json")
    return None, None, None


def preprocess_image(image: Image.Image, target_size=(256, 256)):
    """Convert to RGB, resize to target_size, prepare a batch and preprocess for model."""
    image = image.convert("RGB")
    image = image.resize(target_size)
    arr = np.array(image).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def predict(model, labels, image: Image.Image, input_size=(256, 256), top_k=5):
    """Return sorted list of (label, prob) pairs descending. Accept labels as list or dict."""
    x = preprocess_image(image, target_size=input_size)
    try:
        preds = model.predict(x)[0]
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        return []

    # ensure preds is 1D numpy
    preds = np.array(preds).ravel()

    # normalize if probabilities don't sum to 1 (safety)
    total = preds.sum()
    if total > 0:
        probs = preds / total
    else:
        probs = preds

    # handle labels format: list map idx->name, or dict {idx/name: label}
    if labels is None:
        label_list = [str(i) for i in range(len(probs))]
    elif isinstance(labels, dict):
        # if labels are a mapping of indices -> names or names -> idx, try to produce an ordered list
        # If keys are numeric strings, sort by int key
        try:
            items = sorted(labels.items(), key=lambda kv: int(kv[0]))
            label_list = [v for _, v in items]
        except Exception:
            # fallback: take values in insertion order
            label_list = list(labels.values())
            if len(label_list) != len(probs):
                # fallback to index labels
                label_list = [str(i) for i in range(len(probs))]
    else:
        label_list = list(labels)

    # make pairs and sort
    items = list(zip(label_list, probs.tolist()))
    items.sort(key=lambda t: t[1], reverse=True)
    return items[:top_k]


def main():
    st.set_page_config(page_title="八哥辨識 (Myna Classifier)", layout="centered")
    st.title("八哥辨識 (Myna Classifier)")
    st.write("Upload a photo of a myna and the model will show predicted probabilities.")

    model, labels, input_size = load_model_and_labels()
    if model is None:
        st.warning(
            "Model not found. Please create `model.h5` and `model_labels.json` "
            "or put a model in the `models/` folder."
        )

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    top_k = st.slider("Top K predictions to show", min_value=1, max_value=10, value=5)

    if uploaded is not None and model is not None:
        try:
            # ensure we read from start
            uploaded.seek(0)
        except Exception:
            pass

        image = Image.open(BytesIO(uploaded.read()))
        st.image(image, caption="Uploaded", use_column_width=True)
        st.info("Classifying...")

        with st.spinner("Running model..."):
            items = predict(model, labels, image, input_size=input_size or (256, 256), top_k=top_k)

        if not items:
            st.error("No predictions available.")
            return

        # Display predictions as table and bar chart
        for label, prob in items:
            st.write(f"**{label}**: {prob:.4f}")

        try:
            # bar chart for visualization
            names = [t[0] for t in items]
            probs = [t[1] for t in items]
            chart_data = { "probability": probs }
            st.bar_chart(data={n: [p] for n, p in zip(names, probs)})
        except Exception:
            # chart is optional, ignore failures
            pass


if __name__ == "__main__":
    main()
