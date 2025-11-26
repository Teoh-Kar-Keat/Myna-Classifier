import os
import json
from io import BytesIO

import numpy as np
import pandas as pd
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
# 將可能的 list / ndarray 轉 float
# -------------------------------
def flatten_prob(p):
    while isinstance(p, (list, np.ndarray)):
        if isinstance(p, np.ndarray) and p.shape == ():  # scalar
            break
        p = p[0]
    return float(p)

# -------------------------------
# 預測所有類別
# -------------------------------
def predict_all(model, labels, image: Image.Image):
    x = preprocess_image(image)
    preds = model.predict(x)  # 不先取 [0]，保留原始輸出

    # 檢查原始輸出結構
    st.write("模型原始輸出：", preds, type(preds), preds.shape if isinstance(preds, np.ndarray) else "")

    # 如果是多維度 (batch, num_classes, …)，取第一個 batch 並展平
    if isinstance(preds, list) or (isinstance(preds, np.ndarray) and len(preds.shape) > 1):
        preds = np.array(preds).reshape(-1)

    if labels is None:
        labels = [str(i) for i in range(len(preds))]

    # 中文 label 對照表
    label_map = {
        "common_myna": "家八哥",
        "crested_myna": "八哥",
        "javan_myna": "白尾八哥"
    }

    # 建立 (中文名稱, 機率) 列表
    items = []
    for lbl, p in zip(labels, preds):
        name = label_map.get(lbl, lbl)  # 找不到就用原本 label
        prob = flatten_prob(p)
        items.append((name, prob))

    return items

# -------------------------------
# Streamlit App
# -------------------------------
def main():
    st.set_page_config(page_title="八哥辨識", layout="centered")
    st.title("八哥辨識 (Myna Classifier)")
    st.write("上傳八哥的照片，模型會預測該鳥的種類並顯示所有機率與柱狀圖。")
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
            results = predict_all(model, labels, image)

            # 顯示文字結果
            st.write("### 預測結果")
            for name, prob in results:
                st.write(f"- **{name}**: {prob:.4f}")

            # 顯示柱狀圖
            st.markdown("---")
            st.write("### 機率柱狀圖")
            df = pd.DataFrame({
                "機率": [prob for _, prob in results]
            }, index=[name for name, _ in results])
            st.bar_chart(df)

        except Exception as e:
            st.error(f"預測失敗: {e}")

if __name__ == "__main__":
    main()
