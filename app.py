import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import json
import pandas as pd

# ---------- 模型與標籤載入 ----------
@st.cache_resource
def load_model_and_labels(model_path="model.h5", labels_path="model_labels.json"):
    model = tf.keras.models.load_model(model_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    return model, labels

model, labels = load_model_and_labels()

# ---------- 鳥類介紹資料 ----------
bird_info = {
    "Acridotheres cristatellus formosanus": "中文名 八哥（臺灣）\n綜合描述: 雌雄鳥同色，全身黑色，額部羽毛上豎成羽冠狀…",
    "Acridotheres tristis": "中文名 家八哥\n綜合描述: 全長約25-26cm，頭及尾羽黑色，身體褐色，喙黃色…",
    "Acridotheres javanicus": "中文名 白尾八哥\n概述: 全身灰黑色為主，嘴、腳橘黃色。雜食性，包括種子、水果、昆蟲…"
}

# ---------- 預測函數 ----------
def predict(image):
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)[0]
    return {labels[i]: float(preds[i]) for i in range(len(labels))}

# ---------- Streamlit App ----------
st.set_page_config(page_title="鳥類辨識", layout="wide")
st.title("🔹 超級美化亮點：鳥類辨識系統")

uploaded_file = st.file_uploader("請上傳鳥類圖片", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="上傳的圖片", use_column_width=True)
    
    # 預測
    st.write("正在辨識中...")
    preds = predict(image)
    
    # 顯示所有類別機率
    st.subheader("所有類別機率")
    df = pd.DataFrame(list(preds.items()), columns=["鳥類", "機率"])
    df = df.sort_values("機率", ascending=False)
    
    st.bar_chart(data=df.set_index("鳥類"), width=0, height=300)
    
    # 下拉選單選鳥類
    st.subheader("鳥類詳細介紹")
    bird_choice = st.selectbox("選擇鳥類查看介紹", options=list(bird_info.keys()))
    st.text(bird_info[bird_choice])
