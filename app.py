import os
import json
import time
import random
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import altair as alt

# ------------------------------------------------------
# 安全導入 TensorFlow (若無安裝或載入失敗，自動切換至 Demo 模式)
# ------------------------------------------------------
try:
    import tensorflow as tf
    from tensorflow.keras.applications.resnet_v2 import preprocess_input
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.toast("⚠️ 未偵測到 TensorFlow，將進入 UI 展示模式", icon="🌿")

# ------------------------------------------------------
# 頁面設定與 CSS 生態風格美化
# ------------------------------------------------------
st.set_page_config(
    page_title="野外八哥辨識圖鑑 🦜",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 自定義 CSS：生態系配色 (Earth Tones & Nature Greens)
st.markdown("""
<style>
    /* 全局背景色 - 米黃色紙張感 */
    .stApp {
        background-color: #F9F7F1;
    }
    
    /* 標題樣式 - 森林綠 */
    h1, h2, h3 {
        color: #2F4F4F !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* 強調文字 */
    .highlight-text {
        color: #556B2F;
        font-weight: bold;
    }

    /* 資訊卡片容器 */
    .info-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 15px;
        border-left: 8px solid #8FBC8F; /* 淺綠色邊框 */
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }

    /* 標籤 (Badges) */
    .badge {
        display: inline-block;
        padding: 5px 12px;
        margin: 2px;
        font-size: 14px;
        font-weight: 600;
        border-radius: 15px;
        color: white;
    }
    .badge-native { background-color: #228B22; } /* 綠色：原生/特有 */
    .badge-invasive { background-color: #CD5C5C; } /* 紅色：外來/入侵 */
    .badge-neutral { background-color: #DAA520; } /* 金色：其他 */

    /* 進度條顏色覆蓋 */
    .stProgress > div > div > div > div {
        background-color: #556B2F;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# 資料庫 (Bird Info) - 保持原本內容，增加標籤欄位
# ------------------------------------------------------
bird_info = {
    "家八哥": {
        "學名": "Acridotheres cristatellus formosanus",
        "中文名": "八哥（臺灣亞種）",
        "標籤": ["台灣特有亞種", "珍貴稀有", "原生種"],
        "標籤顏色": ["native", "native", "native"],
        "綜合描述": "雌雄同色，全身黑色，額羽豎立如冠羽。翅上具明顯白色翼斑，尾羽末端白色。虹膜橙黃，喙象牙白色，跗蹠暗黃。",
        "棲地": "生活於海拔 2,100m 以下之竹林、稀疏林地、農地、都市開放空間。",
        "習性": "雜食性，地面覓食昆蟲、種子、水果，常在牛背啄食體外寄生蟲。一年 1–2 次繁殖。",
        "保育狀態": "臺灣紅皮書近危（NT）",
        "威脅": "棲地破壞、人為干擾、外來種競爭（主要受家八哥、白尾八哥威脅）。"
    },
    "common_myna": { # 對應模型標籤名稱，展示時會轉中文
        "中文名": "家八哥",
        "學名": "Acridotheres tristis",
        "標籤": ["外來種", "強勢物種", "入侵風險"],
        "標籤顏色": ["invasive", "invasive", "invasive"],
        "綜合描述": "全身深褐黑色，頭部至上胸較黑。眼周裸皮明顯呈黃色。喙與腳呈亮黃色。叫聲多變、響亮，適應力極強。",
        "棲地": "都市、公園、農田、住家建築附近皆可見，是強勢適應者。",
        "習性": "雜食性，攝食昆蟲、穀物、水果、人類廚餘。一年 1–3 次繁殖。",
        "保育狀態": "全球無危（LC），但在台灣為常見外來種。",
        "威脅": "排擠原生種鳥類，搶奪巢位。"
    },
    "javan_myna": {
        "中文名": "白尾八哥",
        "學名": "Acridotheres javanicus",
        "標籤": ["外來種", "籠鳥逸出", "易危(原產地)"],
        "標籤顏色": ["invasive", "neutral", "neutral"],
        "綜合描述": "體型較小，全身黑色但尾羽末端具明顯白斑。眼周裸皮較不明顯，喙與腳為黃色。",
        "棲地": "都市邊緣、農地、小型林地。",
        "習性": "雜食性，包含昆蟲、水果、穀類。行為敏捷。",
        "保育狀態": "原產地易危（VU），在台灣為外來種。",
        "威脅": "與原生八哥競爭食物與棲地。"
    }
}

# 標籤映射修正 (確保鍵值對應)
LABEL_MAP = {
    "common_myna": "家八哥",
    "crested_myna": "八哥",  # 原生種
    "javan_myna": "白尾八哥"
}

# 反向映射用於查找資料
INFO_KEY_MAP = {
    "家八哥": "common_myna",
    "八哥": "家八哥", # 注意：這裡您的原始資料key是"家八哥"(原生)跟"家八哥"(外來)名字重疊了，我這裡假設 bird_info 的 key 已經調整
    "白尾八哥": "javan_myna"
}

# 修正 bird_info 的 Key 以配合邏輯
bird_info_clean = {
    "八哥": bird_info["家八哥"], # 原生
    "家八哥": bird_info["common_myna"], # 外來
    "白尾八哥": bird_info["javan_myna"] # 外來
}

# ------------------------------------------------------
# 核心邏輯：模型載入與預測 (含 Mock 機制)
# ------------------------------------------------------
@st.cache_resource
def load_model_and_labels(model_path="models/myna_model.keras",
                          labels_path="models/labels.json"):
    
    # 模擬模式判斷
    if not TF_AVAILABLE or not os.path.exists(model_path):
        return "MOCK_MODEL", ["common_myna", "crested_myna", "javan_myna"]

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception:
        return "MOCK_MODEL", ["common_myna", "crested_myna", "javan_myna"]

    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
    else:
        labels = ["common_myna", "crested_myna", "javan_myna"]

    return model, labels

def predict_image(model, labels, image: Image.Image):
    """
    如果 model 是字串 'MOCK_MODEL'，則回傳隨機數據供展示用。
    否則執行真正的預測。
    """
    if model == "MOCK_MODEL":
        # 模擬延遲，增加真實感
        time.sleep(0.8)
        # 產生隨機機率，總和為 1
        probs = np.random.dirichlet(np.ones(len(labels)), size=1)[0]
        # 排序
        results = []
        for lbl, p in zip(labels, probs):
            chi_name = LABEL_MAP.get(lbl, lbl)
            results.append((chi_name, float(p)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # --- 真實預測邏輯 ---
    image = image.convert("RGB").resize((256, 256))
    arr = np.array(image).astype(np.float32)
    if arr.ndim == 2: arr = np.stack([arr]*3, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    
    preds = model.predict(arr)
    if isinstance(preds, list): preds = np.array(preds).reshape(-1)
    else: preds = preds.squeeze()
    if preds.ndim > 1: preds = preds.reshape(-1)
    
    results = []
    for i, p in enumerate(preds):
        lbl = labels[i] if i < len(labels) else str(i)
        chi_name = LABEL_MAP.get(lbl, lbl)
        results.append((chi_name, float(p)))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# ------------------------------------------------------
# UI 主介面
# ------------------------------------------------------
def main():
    # 頂部標題區
    st.markdown("<div style='text-align: center; margin-bottom: 20px;'>", unsafe_allow_html=True)
    st.markdown("<h1>🦜 野外八哥辨識圖鑑</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.2em; color: #556B2F;'>— 上傳照片，透過 AI 辨識您的野外觀察紀錄 —</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 檢查模型狀態
    model, labels = load_model_and_labels()
    
    if model == "MOCK_MODEL":
        st.warning("⚠️ 系統正在使用 **演示模式 (Demo Mode)**。預測結果為隨機生成，僅供版面測試。", icon="🛠️")

    # 兩欄式佈局
    col_img, col_res = st.columns([1, 1.2], gap="large")

    with col_img:
        st.markdown("### 📷 上傳觀察照片")
        uploaded = st.file_uploader("選擇一張 JPG/PNG 圖片", type=["jpg", "jpeg", "png"])
        
        if uploaded:
            image = Image.open(BytesIO(uploaded.read()))
            st.image(image, caption="您的觀察紀錄", use_container_width=True)
            
            # 開始分析按鈕 (增加互動感)
            start_btn = True # 自動開始
        else:
            # 佔位圖 (Placeholder)
            st.markdown(
                """
                <div style='border: 2px dashed #ccc; border-radius: 10px; height: 300px; display: flex; align-items: center; justify-content: center; color: #aaa;'>
                    <span>請上傳圖片以開始分析</span>
                </div>
                """, unsafe_allow_html=True
            )
            start_btn = False

    with col_res:
        if start_btn and uploaded:
            with st.spinner("🔍 正在比對物種特徵..."):
                results = predict_image(model, labels, image)
            
            top_bird, top_prob = results[0]
            
            # --- 1. 結果摘要卡片 ---
            st.markdown(f"""
            <div class='info-card' style='border-left-color: #228B22; background-color: #F0FFF0;'>
                <h2 style='margin:0; color: #006400;'>辨識結果：{top_bird}</h2>
                <p style='font-size: 1.1em; color: #555;'>信心指數：<b>{top_prob*100:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

            # --- 2. 機率圖表 (使用 Altair 優化) ---
            df = pd.DataFrame(results, columns=["物種", "機率"])
            df["機率(%)"] = (df["機率"] * 100).round(1)
            
            chart = alt.Chart(df).mark_bar(cornerRadiusTopRight=10, cornerRadiusBottomRight=10).encode(
                x=alt.X('機率(%)', title=None),
                y=alt.Y('物種', sort='-x', title=None),
                color=alt.Color('機率', scale=alt.Scale(scheme='greens'), legend=None),
                tooltip=['物種', '機率(%)']
            ).properties(height=200, title="AI 預測機率分佈")
            
            st.altair_chart(chart, use_container_width=True)

            # --- 3. 生態圖鑑資料 (Tab 分頁) ---
            info = bird_info_clean.get(top_bird)
            
            if info:
                st.markdown("### 📖 物種圖鑑")
                
                # 標籤顯示
                tags_html = ""
                for tag, color in zip(info.get("標籤", []), info.get("標籤顏色", [])):
                    tags_html += f"<span class='badge badge-{color}'>{tag}</span>"
                st.markdown(f"<div style='margin-bottom:15px;'>{tags_html}</div>", unsafe_allow_html=True)

                # 分頁內容
                tab1, tab2, tab3 = st.tabs(["🌿 基本資料", "🏞️ 棲地與習性", "🛡️ 保育資訊"])
                
                with tab1:
                    st.markdown(f"**學名**：*{info['學名']}*")
                    st.markdown(f"**特徵描述**：<br>{info['綜合描述']}", unsafe_allow_html=True)
                
                with tab2:
                    st.info(f"**棲地環境**：{info['棲地']}")
                    st.success(f"**覓食習性**：{info['習性']}")

                with tab3:
                    st.warning(f"**保育狀態**：{info['保育狀態']}")
                    st.error(f"**生存威脅**：{info['威脅']}")
            else:
                st.info("暫無此物種詳細生態資料。")

        elif not start_btn:
            # 未上傳時的引導文字
            st.markdown("### 💡 如何使用")
            st.markdown("""
            1. 點擊左側 **Browse files** 上傳照片。
            2. 系統將自動進行特徵提取與分類。
            3. 查看右側的物種介紹與保育建議。
            
            *本工具支援：家八哥、白尾八哥及台灣原生八哥。*
            """)

    # 頁腳
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888; font-size: 0.8em;'>Designed for Ecological Education & Citizen Science | Powered by Streamlit</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
