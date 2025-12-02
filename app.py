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
# 安全導入 TensorFlow
# ------------------------------------------------------
try:
    import tensorflow as tf
    from tensorflow.keras.applications.resnet_v2 import preprocess_input
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ------------------------------------------------------
# 頁面設定與 CSS (針對大字體與閱讀性優化)
# ------------------------------------------------------
st.set_page_config(
    page_title="野外八哥辨識圖鑑 🦜",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* 全局背景與字體設定 */
    .stApp { background-color: #F9F7F1; }
    
    /* === 字體放大區 === */
    /* 一般段落文字放大 */
    .big-font {
        font-size: 1.15rem !important;
        line-height: 1.7 !important;
        color: #333333;
        font-family: "Microsoft JhengHei", "Helvetica Neue", sans-serif;
    }
    /* 標題放大 */
    h1 { font-size: 2.5rem !important; color: #2F4F4F !important; }
    h2 { font-size: 2.0rem !important; color: #2F4F4F !important; }
    h3 { font-size: 1.5rem !important; color: #556B2F !important; font-weight: bold !important;}
    
    /* 側邊欄優化 */
    section[data-testid="stSidebar"] { background-color: #E8F3E8; }

    /* 圖片容器限制 */
    .bird-image-container img {
        max-height: 400px !important;
        object-fit: contain;
        border-radius: 12px;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.15);
        border: 5px solid #fff;
    }
    
    /* 資訊卡片樣式 */
    .info-box {
        background-color: #FFFFFF;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border-top: 5px solid #8FBC8F;
    }

    /* 標籤 (Badges) */
    .badge {
        display: inline-block; padding: 6px 14px; margin: 4px;
        font-size: 1rem; font-weight: 600; border-radius: 20px; color: white;
        letter-spacing: 1px;
    }
    .badge-native { background-color: #556B2F; box-shadow: 0 2px 4px rgba(85,107,47,0.4); }
    .badge-invasive { background-color: #CD5C5C; box-shadow: 0 2px 4px rgba(205,92,92,0.4); }
    .badge-neutral { background-color: #DAA520; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# 詳細資料庫 (根據您提供的資料更新)
# ------------------------------------------------------
bird_info = {
    "家八哥": {
        "學名": "Acridotheres cristatellus formosanus",
        "中文名": "八哥（臺灣特有亞種）",
        "標籤": ["臺灣特有亞種", "珍貴稀有(II)", "近危(NT)"],
        "標籤顏色": ["native", "native", "native"],
        "外觀": """
        <ul class="big-font">
            <li><b>整體：</b> 雌雄同色，全身幾為純黑色。</li>
            <li><b>頭部：</b> 額羽聳立於喙基上如冠羽（這一點非常重要）。虹膜橙黃色，喙象牙白色。</li>
            <li><b>翅膀：</b> 翼上有明顯白斑，初級覆羽先端和初級飛羽基部為白色，飛行時非常明顯。</li>
            <li><b>尾部：</b> 尾羽末端為白色，尾下覆羽黑白相間。</li>
            <li><b>腳：</b> 跗蹠暗黃色。</li>
        </ul>
        """,
        "習性": """
        <div class="big-font">
            <p><b>棲地：</b> 生活在海拔 2,100m 以下之竹林、疏林、開闊地區。常見於高速公路護欄、燈架、電線及農田牛背上。</p>
            <p><b>食性：</b> 雜食性。主要在地面覓食，常在耕地啄食蚯蚓、昆蟲、植物塊莖。也會在牛背上啄食體外寄生蟲。</p>
            <p><b>繁殖：</b> 繁殖期 3-7 月，築巢於樹洞、電桿或鐵塔。一季可育兩窩，每窩產 3-5 枚卵（淡藍色或藍綠色）。</p>
            <p><b>行為：</b> 具群聚性，清晨傍晚常聚大群。會模仿環境聲音及人語。</p>
        </div>
        """,
        "保育": """
        <div class="big-font">
            <p><b>狀態：</b> 台灣紅皮書列為「近危 (NT)」。野生動物保育法公告之「珍貴稀有野生動物」。</p>
            <p><b>威脅：</b> 受到外來種八哥（家八哥、白尾八哥）的強勢競爭，巢位與食物資源被搶奪，導致野外數量快速減少。</p>
            <p><b>法規：</b> 屬第二級珍貴稀有保育類，受法律保護。</p>
        </div>
        """
    },
    "common_myna": {
        "中文名": "家八哥",
        "學名": "Acridotheres tristis",
        "標籤": ["外來入侵種", "全球百大入侵種", "強勢物種"],
        "標籤顏色": ["invasive", "invasive", "invasive"],
        "外觀": """
        <ul class="big-font">
            <li><b>整體：</b> 全長約 25-26cm，身體褐色，頭及喉部黑色。</li>
            <li><b>頭部：</b> <b>眼周裸皮明顯呈黃色</b>（這是最明顯特徵），喙與腳呈亮黃色。無額前冠羽。</li>
            <li><b>翅膀：</b> 飛行時可見明顯的白色翼斑。</li>
            <li><b>尾部：</b> 尾羽黑色，末端白色。</li>
        </ul>
        """,
        "習性": """
        <div class="big-font">
            <p><b>棲地：</b> 極度適應人類環境。遍布都市公園、校園、農地、垃圾場。</p>
            <p><b>食性：</b> 雜食性且機會主義者。昆蟲、果實、廚餘垃圾、小型脊椎動物皆吃。</p>
            <p><b>繁殖：</b> 營穴巢，利用建築物縫隙、招牌、路標管洞築巢。繁殖力強，排擠原生鳥類。</p>
            <p><b>行為：</b> 領域性強，噪鳴聲響亮且多變。極不怕人，常成群活動。</p>
        </div>
        """,
        "保育": """
        <div class="big-font">
            <p><b>風險：</b> IUCN 全球百大入侵種之一。與原生八哥競爭巢位與食物，甚至捕食原生鳥類的蛋與雛鳥。</p>
            <p><b>狀態：</b> 在台灣為強勢外來種，無保育等級，需進行族群控制。</p>
            <p><b>傳播：</b> 早期因能模仿人語而被大量引入作為寵物，後逃逸或放生擴散。</p>
        </div>
        """
    },
    "javan_myna": {
        "中文名": "白尾八哥",
        "學名": "Acridotheres javanicus",
        "標籤": ["外來入侵種", "台灣數量最多", "原產地易危"],
        "標籤顏色": ["invasive", "invasive", "neutral"],
        "外觀": """
        <ul class="big-font">
            <li><b>整體：</b> 全身灰黑色為主，體型約 21-23cm。</li>
            <li><b>頭部：</b> 有短羽冠（不如原生八哥明顯），喙與腳為橘黃色。虹膜橘黃（幼鳥灰白）。</li>
            <li><b>尾部：</b> <b>尾羽末端及尾下覆羽為白色</b>，因此得名「白尾八哥」。</li>
            <li><b>區別：</b> 與家八哥不同處在於全身偏灰黑且無眼周裸皮；與原生八哥不同處在於喙是黃色（原生為象牙白）且體色較灰。</li>
        </ul>
        """,
        "習性": """
        <div class="big-font">
            <p><b>棲地：</b> 平原、近郊丘陵、都市草地。目前是台灣數量最多的外來八哥。</p>
            <p><b>食性：</b> 雜食性。喜愛在剛割完草的草地覓食昆蟲，也會吃人類垃圾。</p>
            <p><b>繁殖：</b> 適應力極強，利用都市建築縫隙築巢。</p>
            <p><b>行為：</b> 性情兇悍，常驅趕麻雀或其他鳥類。夜間有集體夜棲習性，數量可達上百隻。</p>
        </div>
        """,
        "保育": """
        <div class="big-font">
            <p><b>風險：</b> 嚴重排擠原生八哥生存空間。在台灣野外已建立穩定且龐大的族群。</p>
            <p><b>狀態：</b> 國際自然保育聯盟(IUCN)在其原產地列為易危(VU)，但在台灣是需要控制的入侵種。</p>
            <p><b>來源：</b> 1978年首次紀錄，主要由籠鳥逃逸或宗教放生導致擴散。</p>
        </div>
        """
    }
}

# 標籤映射
LABEL_MAP = { "common_myna": "家八哥", "crested_myna": "八哥", "javan_myna": "白尾八哥" }
bird_info_clean = { "八哥": bird_info["家八哥"], "家八哥": bird_info["common_myna"], "白尾八哥": bird_info["javan_myna"] }

# ------------------------------------------------------
# 核心邏輯 (保持 Mock 模式以利展示)
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
        time.sleep(0.6)
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
# UI 主介面
# ------------------------------------------------------
def main():
    model, labels = load_model_and_labels()

    # --- 側邊欄 ---
    with st.sidebar:
        st.markdown("## 📷 觀察紀錄上傳")
        st.markdown("請上傳您拍攝到的八哥照片：")
        uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"])
        
        st.markdown("---")
        st.markdown("### 辨識支援")
        st.markdown("- **八哥 (原生種)**\n- **家八哥 (外來種)**\n- **白尾八哥 (外來種)**")
        if model == "MOCK_MODEL":
            st.info("目前為演示模式 (Demo Mode)")

    # --- 主畫面標題 ---
    st.markdown("<h1>🦜 野外八哥辨識圖鑑</h1>", unsafe_allow_html=True)
    st.markdown("<p class='big-font' style='color:#666;'>上傳照片，AI 將協助您辨識物種，並提供詳細的生態保育資訊。</p>", unsafe_allow_html=True)
    st.markdown("---")

    if not uploaded:
        st.warning("👈 請從左側側邊欄上傳圖片以開始分析")
        return

    # --- 分析結果區 (上圖下文結構) ---
    col_img, col_stat = st.columns([0.8, 1.2], gap="large")

    image = Image.open(BytesIO(uploaded.read()))

    with col_img:
        st.markdown('<div class="bird-image-container">', unsafe_allow_html=True)
        st.image(image, caption="您的觀察照片", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_stat:
        # 執行辨識
        with st.spinner("正在分析物種特徵..."):
            results = predict_image(model, labels, image)
        
        top_bird, top_prob = results[0]
        
        # 1. 結果標題與信心度
        st.markdown(f"""
        <div style="background-color: white; padding: 20px; border-radius: 10px; border-left: 8px solid #2F4F4F; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <h2 style="margin:0;">{top_bird}</h2>
            <p class="big-font" style="margin-bottom:0; color: #556B2F;">信心指數：<b>{top_prob*100:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        # 2. 標籤顯示
        info = bird_info_clean.get(top_bird)
        if info:
            st.markdown("<div style='margin-top: 15px;'>", unsafe_allow_html=True)
            for t, c in zip(info.get("標籤", []), info.get("標籤顏色", [])):
                st.markdown(f"<span class='badge badge-{c}'>{t}</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown(f"<p class='big-font'><b>學名：</b> <i>{info.get('學名')}</i></p>", unsafe_allow_html=True)

        # 3. 機率圖表 (精簡化)
        df = pd.DataFrame(results, columns=["物種", "機率"])
        df["機率(%)"] = (df["機率"] * 100).round(1)
        
        base = alt.Chart(df).encode(y=alt.Y('物種', sort='-x', title=None))
        bar = base.mark_bar(color="#8FBC8F").encode(x=alt.X('機率(%)', title=None))
        text = base.mark_text(align='left', dx=5).encode(x='機率(%)', text='機率(%)')
        
        st.altair_chart((bar + text).properties(height=130), use_container_width=True)

    # --- 詳細資訊區 (Tabs 分頁設計) ---
    st.markdown("### 📖 物種詳細檔案")
    
    if info:
        # 使用 Tabs 來整理大量資訊
        tab1, tab2, tab3 = st.tabs(["🔍 外觀與特徵", "🌿 生態與習性", "🛡️ 保育與分佈"])
        
        with tab1:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown(info['外觀'], unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with tab2:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown(info['習性'], unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with tab3:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown(info['保育'], unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("暫無詳細資料")

if __name__ == "__main__":
    main()
