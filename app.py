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

# ------------------------------------------------------
# 頁面設定 + CSS（完全左右對稱）
# ------------------------------------------------------
st.set_page_config(page_title="八哥辨識器 🦜", layout="wide")

page_css = """
<style>
body {
    background-image: linear-gradient(to bottom right, #f0f8ff, #e6e6fa);
}
.stApp {
    color: #4B0082;
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
}

/* 兩欄卡片：左右完全對稱 */
.card-box {
    background: white;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 2px 2px 15px rgba(0,0,0,0.15);
    min-height: 520px;     /* → 左右同高度 */
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
}

/* 圖片置中 */
.card-box img {
    display: block;
    margin-left: auto;
    margin-right: auto;
}
</style>
"""
st.markdown(page_css, unsafe_allow_html=True)

# ------------------------------------------------------
# 模型與標籤載入（不修改你的原邏輯）
# ------------------------------------------------------
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
        labels = None
    else:
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                labels = json.load(f)
        except:
            labels = None

    return model, labels

# ------------------------------------------------------
# 圖片預處理（保留）
# ------------------------------------------------------
def preprocess_image(image: Image.Image, target_size=(256, 256)):
    image = image.convert("RGB")
    image = image.resize((target_size[1], target_size[0]))
    arr = np.array(image).astype(np.float32)

    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)

    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

# ------------------------------------------------------
# 預測（保留）
# ------------------------------------------------------
def predict_all(model, labels, image: Image.Image):
    x = preprocess_image(image)
    preds = model.predict(x)

    if isinstance(preds, list):
        preds = np.array(preds).reshape(-1)
    else:
        preds = preds.squeeze()
        if preds.ndim > 1:
            preds = preds.reshape(-1)

    if labels is None:
        labels = [str(i) for i in range(len(preds))]

    label_map = {
        "common_myna": "家八哥",
        "crested_myna": "八哥",
        "javan_myna": "白尾八哥"
    }

    return [(label_map.get(lbl, lbl), float(prob)) for lbl, prob in zip(labels, preds)]

# ------------------------------------------------------
# UI 主介面（只改 CSS class）
# ------------------------------------------------------
def main():

    st.markdown("<h1 style='text-align:center; font-size:50px;'>🦜 八哥辨識器</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:20px;'>上傳八哥圖片，即可獲得分類與機率分析</p>", unsafe_allow_html=True)
    st.markdown("---")

    model, labels = load_model_and_labels()
    if model is None:
        return

    col1, col2 = st.columns(2, gap="large")

    # 左邊
    with col1:
        st.markdown("<div class='card-box'>", unsafe_allow_html=True)

        uploaded = st.file_uploader("📂 上傳八哥圖片", type=["jpg", "jpeg", "png"])
        image = None

        if uploaded:
            image = Image.open(BytesIO(uploaded.read()))
            st.image(image, caption="已上傳圖片", width=320)

        st.markdown("</div>", unsafe_allow_html=True)

    # 右邊
    with col2:
        st.markdown("<div class='card-box'>", unsafe_allow_html=True)

        if uploaded and image is not None:

            st.markdown("### 🔍 預測結果")

            results = predict_all(model, labels, image)
            results.sort(key=lambda x: x[1], reverse=True)

            # 機率卡片
            for i, (name, prob) in enumerate(results):
                color = "#32CD32" if i == 0 else "#87CEFA"
                st.markdown(
                    f"""
                    <div style='background-color:{color}; 
                                padding:12px;
                                border-radius:15px;
                                margin-bottom:8px;
                                box-shadow:2px 2px 5px rgba(0,0,0,0.2);'>
                        <h4 style='color:white; margin:0;'>{name}: {prob*100:.2f}%</h4>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Altair 柱狀圖
            df = pd.DataFrame({
                "類別": [name for name, _ in results],
                "機率": [prob*100 for _, prob in results]
            })

            chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X("機率", title="機率 (%)"),
                    y=alt.Y("類別", sort='-x', title="八哥種類"),
                    color=alt.condition(
                        alt.datum.機率 == df["機率"].max(),
                        alt.value("green"),
                        alt.value("skyblue")
                    ),
                    tooltip=["類別", "機率"]
                )
                .properties(height=250)
            )
            st.altair_chart(chart, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
