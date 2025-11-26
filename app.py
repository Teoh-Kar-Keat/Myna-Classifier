import os
import json
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# -------------------------------
# 模型與標籤載入
# -------------------------------
@st.cache_resource
def load_model_and_labels(
    model_path="models/myna_model.keras",
    labels_path="models/labels.json"
):
    if not os.path.exists(model_path):
        st.error(f"模型檔案不存在：{model_path}")
        return None, None

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"載入模型失敗: {e}")
        return None, None

    if not os.path.exists(labels_path):
        st.warning(f"Labels 檔案不存在，將使用索引標籤。")
        labels = None
    else:
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                labels = json.load(f)
        except Exception as e:
            st.warning(f"讀取 labels 失敗: {e}")
            labels = None

    return model, labels

# -------------------------------
# 圖片預處理
# -------------------------------
def preprocess_image(image: Image.Image, target_size=(256, 256)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    arr = np.array(image).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

# -------------------------------
# 預測
# -------------------------------
def predict(model, labels, image: Image.Image, top_k=5):
    x = preprocess_image(image)
    preds = model.predict(x)[0]  # 取 batch 的第一個結果

    if labels is None:
        labels = [str(i) for i in range(len(preds))]

    # 確保 prob 是 float
    items = [(lbl, float(p)) for lbl, p in zip(labels, preds.tolist())]
    items.sort(key=lambda t: t[1], reverse=True)
    return items[:top_k]

# -------------------------------
# Streamlit App
# -------------------------------
def main():
    st.set_page_config(page_title="八哥辨識", layout="centered")
    st.title("八哥辨識 (Myna Classifier)")
    st.write("上傳八哥的照片，模型會預測該鳥的種類並顯示機率。")
    st.markdown("---")

    # 載入模型與 labels
    model, labels = load_model_and_labels()
    if model is None:
        st.warning("找不到模型，請先運行 training.py 產生模型與 labels.json，然後重新整理此頁面。")
        return

    # 上傳圖片
    uploaded = st.file_uploader("選擇圖片", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        try:
            image = Image.open(BytesIO(uploaded.read()))
            st.image(image, caption="已上傳圖片", use_column_width=True)
            st.markdown("---")
        except Exception as e:
            st.error(f"讀取圖片錯誤: {e}")
            return

        st.write("正在辨識中...")
        try:
            results = predict(model, labels, image)
            st.write("### 預測結果 Top 5")
            for label, prob in results:
                st.write(f"- **{label}**: {prob:.4f}")
        except Exception as e:
            st.error(f"預測失敗: {e}")

if __name__ == "__main__":
    main()
