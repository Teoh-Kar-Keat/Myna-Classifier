import streamlit as st
from PIL import Image
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input


@st.cache_resource
def load_model(path="models/myna_model"):
    import os
    from tensorflow.keras.layers import TFSMLayer

    # if a single-file Keras model exists, prefer it
    for ext in (".keras", ".h5"):
        p = path if path.lower().endswith(ext) else path + ext
        if os.path.exists(p):
            return tf.keras.models.load_model(p)

    # if a TF SavedModel directory exists, wrap it using TFSMLayer
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "saved_model.pb")):
        # Create a tiny Keras model that delegates to the SavedModel for inference
        layer = TFSMLayer(path, call_endpoint="serving_default")
        inp = tf.keras.Input(shape=(256, 256, 3), dtype=tf.float32)
        out = layer(inp)
        model = tf.keras.Model(inputs=inp, outputs=out)
        return model

    raise FileNotFoundError(f"No compatible model found at {path} (.keras/.h5 or SavedModel dir)")


@st.cache_data
def load_labels(path="models/labels.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_image(img: Image.Image, model, labels):
    img = img.convert("RGB").resize((256, 256))
    arr = np.array(img)
    proc = preprocess_input(np.expand_dims(arr, 0))
    preds = model.predict(proc).flatten()
    return {labels[i]: float(preds[i]) for i in range(len(labels))}


def main():
    st.title("Myna Classifier (Streamlit)")

    st.markdown("Upload an image of a myna and the model will predict which species it is.")

    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    labels = load_labels()
    model = load_model()

    if uploaded is not None:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded image", use_column_width=True)

        with st.spinner("Running inference..."):
            results = predict_image(img, model, labels)

        # display only the three expected classes in a fixed order, formatted to 3 decimals
        order = ["common_myna", "javan_myna", "crested_myna"]
        st.subheader("Predictions")
        for name in order:
            prob = results.get(name, 0.0)
            st.write(f"{name}: {prob:.3f}")


if __name__ == "__main__":
    main()
