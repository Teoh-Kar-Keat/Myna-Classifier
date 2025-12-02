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
# 安全導入 TensorFlow (保持原本邏輯)
# ------------------------------------------------------
try:
    import tensorflow as tf
    from tensorflow.keras.applications.resnet_v2 import preprocess_input
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ------------------------------------------------------
# 頁面設定與 CSS 生態風格美化 (優化版)
# ------------------------------------------------------
st.set_page_config(
    page_title="野外八哥辨識圖鑑 🦜",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded" # 預設展開側邊欄
)

# 自定義 CSS：生態系配色 + 圖片限制 + 卡片陰影
st.markdown("""
<style>
    /* 全局背景 */
    .stApp { background-color: #F9F7F1; }
    
    /* 側邊欄優化 */
    section[data-testid="stSidebar"] {
        background-color: #E8F3E8; /* 淺綠色背景 */
    }

    /* 圖片容器限制：讓圖片不要無限長，增加陰影與圓角 */
    .bird-image-container img {
        max-height: 450px !important; /* 強制限制最大高度 */
        object-fit: contain; /* 保持比例 */
        border-radius: 10px;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.15); /* 相片陰影 */
        border: 4px solid #fff; /* 白邊相框感 */
    }
    
    /* 標題與文字 */
    h1, h2, h3 { color: #2F4F4F !important; font-family: 'Helvetica Neue', sans-serif; }
    
    /* 資訊卡片 */
    .info-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #8FBC8F;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }

    /* 標籤 (Badges) */
    .badge {
        display: inline-block; padding: 4px 10px; margin: 2px;
        font-size: 13px; font-weight: 600; border-radius: 12px; color: white;
    }
    .badge-native { background-color: #556B2F; }
    .badge-invasive { background-color: #CD5C5C; }
    .badge-neutral { background-color: #DAA520; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# 資料庫 (Bird Info) - 保持不變
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
    "common_myna": {
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

LABEL_MAP = { "common_myna": "家八哥", "crested_myna": "八哥", "javan_myna": "白尾八哥" }
bird_info_clean = { "八哥": bird_info["家八哥"], "家八哥": bird_info["common_myna"], "白尾八哥": bird_info["javan_myna"] }

# ------------------------------------------------------
# 核心邏輯 (保持不變)
# ------------------------------------------------------
@st.cache_resource
def load_model_and_labels(model_path="models/myna_model.keras", labels_path="models/labels.json"):
    if not TF_AVAILABLE or not os.path.exists(model_path):
        return "MOCK_MODEL", ["common_myna", "crested_myna", "javan_myna"]
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception:
        return "MOCK_MODEL", ["common_myna", "crested_myna", "javan_myna"]
    
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f: labels = json.load(f)
    else: labels = ["common_myna", "crested_myna", "javan_myna"]
    return model, labels

def predict_image(model, labels, image: Image.Image):
    if model == "MOCK_MODEL":
        time.sleep(0.5)
        probs = np.random.dirichlet(np.ones(len(labels)), size=1)[0]
        results = [(LABEL_MAP.get(lbl, lbl), float(p)) for lbl, p in zip(labels, probs)]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    image_processed = image.convert("RGB").resize((256, 256))
    arr = np.array(image_processed).astype(np.float32)
    if arr.ndim == 2: arr = np.stack([arr]*3, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    
    preds = model.predict(arr)
    if isinstance(preds, list): preds = np.array(preds).reshape(-1)
    else: preds = preds.squeeze()
    if preds.ndim > 1: preds = preds.reshape(-1)
    
    results = [(LABEL_MAP.get(lbl, lbl), float(p)) for i, p, lbl in zip(range(len(preds)), preds, labels)]
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# ------------------------------------------------------
# UI 主介面 (Layout 重構)
# ------------------------------------------------------
def main():
    # 載入模型
    model, labels = load_model_and_labels()

    # --- 側邊欄：功能操作區 ---
    with st.sidebar:
        st.header("🦜 觀察站操作台")
        st.markdown("請在此上傳您拍攝到的八哥照片，系統將自動進行辨識。")
        
        uploaded = st.file_uploader("📂 上傳照片 (JPG/PNG)", type=["jpg", "jpeg", "png"])
        
        st.markdown("---")
        st.markdown("**支援物種：**")
        st.markdown("- 家八哥 (外來)")
        st.markdown("- 白尾八哥 (外來)")
        st.markdown("- 八哥 (台灣特有亞種)")
        
        if model == "MOCK_MODEL":
            st.warning("⚠️ 演示模式：數據為隨機生成")

    # --- 主畫面：標題 ---
    st.markdown("## 🌿 野外八哥辨識圖鑑")
    
    if not uploaded:
        # 歡迎畫面
        st.info("👈 請從左側側邊欄上傳圖片以開始分析")
        st.markdown("""
        <div style='text-align: center; padding: 50px; color: #888;'>
            <h3>等待觀察紀錄...</h3>
            <p>上傳後，您的照片與分析報告將顯示於此。</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # --- 主畫面：分析結果 (左右佈局調整) ---
    # 這裡將比例改為 [4, 5]，左邊放圖，右邊放主要資訊，比較平衡
    col_img, col_info = st.columns([0.8, 1.2], gap="large")

    image = Image.open(BytesIO(uploaded.read()))

    with col_img:
        # 使用 CSS class 限制圖片高度，並增加相框感
        st.markdown('<div class="bird-image-container">', unsafe_allow_html=True)
        st.image(image, caption="您的觀察照片", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_info:
        # 執行預測
        with st.spinner("正在比對特徵資料庫..."):
            results = predict_image(model, labels, image)
        
        top_bird, top_prob = results[0]
        
        # 結果標題區 (使用 Flexbox 讓結果跟機率並排)
        st.markdown(f"""
        <div style="display: flex; align-items: baseline; justify-content: space-between; border-bottom: 2px solid #8FBC8F; padding-bottom: 10px; margin-bottom: 20px;">
            <div style="font-size: 32px; font-weight: bold; color: #2F4F4F;">{top_bird}</div>
            <div style="font-size: 20px; color: #556B2F;">信心指數: <b>{top_prob*100:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

        # 機率條形圖 (縮減高度，使其不搶戲)
        df = pd.DataFrame(results, columns=["物種", "機率"])
        df["機率(%)"] = (df["機率"] * 100).round(1)
        
        chart = alt.Chart(df).mark_bar(color="#8FBC8F", cornerRadiusEnd=5).encode(
            x=alt.X('機率(%)', title=None),
            y=alt.Y('物種', sort='-x', title=None),
            tooltip=['物種', '機率(%)'],
            text=alt.Text('機率(%)') # 直接在條形圖上顯示數字
        ).properties(height=120) # 降低圖表高度
        
        # 疊加文字標籤
        text = chart.mark_text(align='left', dx=2, color='black').encode(text='機率(%)')
        st.altair_chart(chart + text, use_container_width=True)

    # --- 下方：詳細生態卡片 (全寬度) ---
    st.markdown("---")
    
    info = bird_info_clean.get(top_bird)
    if info:
        # 標籤區
        tags_html = "".join([f"<span class='badge badge-{c}'>{t}</span>" for t, c in zip(info.get("標籤", []), info.get("標籤顏色", []))])
        
        # 使用三欄呈現重點資訊，比 Tab 更直觀
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**📝 基本資料**<br>{tags_html}<br><br>學名：*{info['學名']}*", unsafe_allow_html=True)
        with c2:
            st.markdown(f"**🏞️ 棲地與習性**<br>{info['棲地']}", unsafe_allow_html=True)
        with c3:
            st.markdown(f"**🛡️ 保育與威脅**<br>{info['保育狀態']}<br><span style='color:#CD5C5C'>{info['威脅']}</span>", unsafe_allow_html=True)
            
        # 詳細描述放在最底下的摺疊區，節省空間
        with st.expander("📖 查看完整物種描述"):
            st.write(info['綜合描述'])
            st.write(f"**習性補充：** {info['習性']}")
    else:
        st.info("暫無詳細資料")

if __name__ == "__main__":
    main()
