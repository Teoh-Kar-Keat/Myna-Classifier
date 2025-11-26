import streamlit as st
from PIL import Image
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from pathlib import Path
import os

@st.cache_resource
def load_model(path=None):
    """
    Robust model loader:
    - resolves path relative to this file
    - prints diagnostics to the app
    - supports .keras/.h5 single-file, SavedModel directory, or remote download via MODEL_URL in secrets
    """
    base = Path(__file__).resolve().parent
    models_dir = base / "models"

    # default path if none provided
    if path is None:
        path = models_dir / "myna_model.keras"
    else:
        path = Path(path)

    # Diagnostics: show paths and directory listing in the Streamlit app
    st.write("Working directory:", os.getcwd())
    st.write("App file:", str(Path(__file__).resolve()))
    st.write("Looking for model at:", str(path))
    st.write("models dir exists:", models_dir.exists())
    if models_dir.exists():
        try:
            st.write("models directory contents:", sorted([p.name for p in models_dir.iterdir()]))
        except Exception as e:
            st.write("Could not list models directory:", e)

    # Check for single-file Keras models (.keras or .h5)
    for ext in (".keras", ".h5"):
        candidate = path if str(path).lower().endswith(ext) else path.with_suffix(ext)
        if candidate.exists():
            st.write(f"Loading Keras model from: {candidate}")
            return tf.keras.models.load_model(str(candidate))

    # Check for SavedModel dir
    if path.is_dir() and (path / "saved_model.pb").exists():
        st.write(f"Loading SavedModel from directory: {path}")
        try:
            from tensorflow.keras.layers import TFSMLayer
            layer = TFSMLayer(str(path), call_endpoint="serving_default")
            inp = tf.keras.Input(shape=(256, 256, 3), dtype=tf.float32)
            out = layer(inp)
            model = tf.keras.Model(inputs=inp, outputs=out)
            return model
        except Exception as e:
            st.write("Failed to wrap SavedModel with TFSMLayer:", e)
            raise

    # If missing, attempt to download from MODEL_URL if provided in secrets
    model_url = st.secrets.get("MODEL_URL") if hasattr(st, "secrets") else None
    if model_url:
        st.write("MODEL_URL found in secrets — attempting to download model to /tmp/myna_model.keras")
        tmp_path = Path("/tmp/myna_model.keras")
        try:
            import requests
            if not tmp_path.exists():
                with requests.get(model_url, stream=True) as r:
                    r.raise_for_status()
                    with open(tmp_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024*1024):
                            if chunk:
                                f.write(chunk)
            st.write("Downloaded model to", str(tmp_path))
            return tf.keras.models.load_model(str(tmp_path))
        except Exception as e:
            st.write("Failed to download or load model from MODEL_URL:", e)

    # Final helpful error
    found = []
    if models_dir.exists():
        try:
            found = sorted([p.name for p in models_dir.iterdir()])
        except Exception:
            found = "<could not list>"
    raise FileNotFoundError(
        "No compatible model found. Searched for .keras/.h5 files and SavedModel dir at "
        f"{path}. models dir contents: {found}. If you use Git LFS, ensure LFS files were fetched on deploy. "
        "You can also set a MODEL_URL in Streamlit secrets to download the model at startup."
    )

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

    # load labels (this will raise a clear error here if labels.json doesn't exist)
    try:
        labels = load_labels()
    except FileNotFoundError as e:
        st.error(f"labels.json not found: {e}")
        st.stop()

    # load model and show clear error message if missing
    try:
        model = load_model()
    except FileNotFoundError as e:
        st.error(f"Model load failed: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error loading model: {e}")
        st.stop()

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
