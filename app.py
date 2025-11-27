import os
import json
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import altair as alt

import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# -------------------------------
# 模型與標籤載入
# -------------------------------
@st.cache_resource
def load_model_and_labels(model_path="models/myna_model.keras",
                          labels_path="models/labels.json"):
    if not os.path.exists(model_path):
        st.error(f"模型檔案不存在：{model_path}")
        return None, None
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"載入模型失敗: {e}")
        return None, None

    if not os.path.exists(labels_path):
        st.warning("Labels 不存在，將用索引代替")
        labels = None
    else:
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                labels = json.load(f)
        except Exception as e:
            st.warning(f"Labels 讀取失敗: {e}")
            labels = None
    return model, labels

# -------------------------------
# 圖片預處理
# -------------------------------
def preprocess_image(image: Image.Image, target_size=(256, 256)):
    image = image.convert("RGB")
    image = image.resize((target_size[1], target_size[0]))
    arr = np.array(image).astype(np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

def flatten_prob(p):
    while isinstance(p, (list, np.ndarray)):
        if isinstance(p, np.ndarray) and p.shape == ():
            break
        p = p[0]
    return float(p)

# -------------------------------
# 預測
# -------------------------------
def predict_all(model, labels, image: Image.Image):
    x = preprocess_image(image)
    preds = model.predict(x)

    if isinstance(preds, list):
        preds = np.array(preds).reshape(-1)
    elif isinstance(preds, np.ndarray):
        preds = preds.squeeze()
        if preds.ndim == 0:
            preds = np.array([preds])
        elif preds.ndim > 1:
            preds = preds.reshape(-1)

    if labels is None:
        labels = [str(i) for i in range(len(preds))]

    label_map = {
        "common_myna": "家八哥",
        "crested_myna": "八哥",
        "javan_myna": "白尾八哥"
    }

    items = []
    for lbl, p in zip(labels, preds):
        name = label_map.get(lbl, lbl)
        prob = flatten_prob(p)
        items.append((name, prob))
    return items

# -------------------------------
# 超級美化版 UI
# -------------------------------
def main():
    # ----------------- 背景 & 漸層 -----------------
    page_bg_img = """
    <style>
    body {
        background-image: linear-gradient(to bottom right, #f0f8ff, #e6e6fa);
    }
    .stApp {
        color: #4B0082;
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.set_page_config(page_title="八哥辨識器 🦜", layout="wide")
    
    # ----------------- 頂部標題 -----------------
    st.markdown("<h1 style='text-align:center; color:#4B0082; font-size:50px;'>🦜 八哥辨識器</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:20px;'>上傳八哥圖片，立即預測種類並顯示機率！</p>", unsafe_allow_html=True)
    st.markdown("---")

    # ----------------- 載入模型 -----------------
    model, labels = load_model_and_labels()
    if model is None:
        st.warning("請先建立模型和 labels.json")
        return

    col1, col2 = st.columns([1, 1])

    # ----------------- 左側: 圖片上傳 -----------------
    with col1:
        uploaded = st.file_uploader("📂 上傳八哥圖片", type=["jpg","jpeg","png"])
        if uploaded is not None:
            try:
                image = Image.open(BytesIO(uploaded.read()))
                st.image(image, caption="已上傳圖片", use_container_width=False, width=250, output_format="JPEG")
            except Exception as e:
                st.error(f"圖片讀取錯誤: {e}")
                return

    # ----------------- 右側: 預測結果 -----------------
    with col2:
        if uploaded is not None:
            st.markdown("### 🔍 預測結果")
            try:
                results = predict_all(model, labels, image)
                results.sort(key=lambda x: x[1], reverse=True)

                # 卡片式機率顯示
                for i, (name, prob) in enumerate(results):
                    color = "#32CD32" if i == 0 else "#87CEFA"  # 第一名綠色，其餘藍色
                    st.markdown(f"""
                    <div style='background-color:{color}; padding:12px; border-radius:15px; margin-bottom:8px; box-shadow:2px 2px 5px rgba(0,0,0,0.2);'>
                        <h3 style='color:white; margin:0; padding:0;'>{name}: {prob*100:.2f}%</h3>
                    </div>
                    """, unsafe_allow_html=True)

                # Altair 柱狀圖
                df = pd.DataFrame({
                    "類別": [name for name, _ in results],
                    "機率": [prob*100 for _, prob in results]
                })
                chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X("機率", title="機率 (%)"),
                    y=alt.Y("類別", sort='-x', title="八哥種類"),
                    color=alt.condition(
                        alt.datum.機率 == df['機率'].max(),
                        alt.value("green"),
                        alt.value("skyblue")
                    ),
                    tooltip=["類別", "機率"]
                ).properties(height=250)
                st.altair_chart(chart, use_container_width=True)

            except Exception as e:
                st.error(f"預測失敗: {e}")

if __name__ == "__main__":
    main()
