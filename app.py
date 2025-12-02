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
# 安全導入 TensorFlow (保持 Mock 機制)
# ------------------------------------------------------
try:
    import tensorflow as tf
    from tensorflow.keras.applications.resnet_v2 import preprocess_input
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ------------------------------------------------------
# 頁面設定與 CSS (重點修改區：字體與大小)
# ------------------------------------------------------
st.set_page_config(
    page_title="野外八哥辨識圖鑑 🦜",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* 1. 全局字體設定：優先 Times New Roman (英數)，備選 標楷體 (中文) */
    html, body, [class*="css"] {
        font-family: "Times New Roman", "KaiTi", "DFKai-SB", "BiauKai", serif !important;
    }

    /* 背景色：米色紙張質感 */
    .stApp { background-color: #F9F7F1; }
    
    /* 2. 標題樣式放大 */
    h1 {
        font-size: 3.5rem !important; /* 特大標題 */
        font-weight: bold !important;
        color: #2F4F4F !important; /* 深森林綠 */
        margin-bottom: 0.5em !important;
    }
    h2 {
        font-size: 2.2rem !important;
        font-weight: bold !important;
        color: #2F4F4F !important;
        border-bottom: 2px solid #8FBC8F; /* 標題下裝飾線 */
        padding-bottom: 10px;
    }
    h3 {
        font-size: 1.6rem !important;
        font-weight: bold !important;
        color: #556B2F !important; /* 橄欖綠 */
    }

    /* 3. 內文大字體優化 (.big-font 類別) */
    .big-font {
        font-size: 1.35rem !important; /* 字體加大 */
        line-height: 1.8 !important;    /* 行距加寬，提升閱讀舒適度 */
        color: #222222;                 /* 深灰黑色，比純黑柔和 */
        font-weight: 500;
        text-align: justify;            /* 左右對齊 */
    }
    
    /* 列表優化 */
    .big-font ul, .big-font ol {
        padding-left: 1.5em;
    }
    .big-font li {
        margin-bottom: 12px; /* 列表項目間距 */
    }
    .big-font b {
        color: #006400; /* 粗體字改為深綠色強調 */
    }

    /* 4. 分頁 (Tabs) 字體放大 */
    .stTabs [data-baseweb="tab"] {
        font-size: 1.3rem !important;
        font-family: "Times New Roman", "KaiTi", serif !important;
        font-weight: bold;
    }

    /* 側邊欄背景 */
    section[data-testid="stSidebar"] { background-color: #E8F3E8; }

    /* 圖片容器：相框效果 */
    .bird-image-container img {
        max-height: 450px !important;
        object-fit: contain;
        border-radius: 8px;
        box-shadow: 8px 8px 20px rgba(0,0,0,0.2);
        border: 8px solid #fff;
    }
    
    /* 資訊卡片容器 */
    .info-box {
        background-color: #FFFFFF;
        padding: 30px; /* 內距加大 */
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 25px;
        border-top: 6px solid #8FBC8F;
    }

    /* 標籤 (Badges) */
    .badge {
        display: inline-block; padding: 6px 16px; margin: 5px;
        font-size: 1.1rem; /* 標籤字體加大 */
        font-weight: bold; border-radius: 20px; color: white;
        letter-spacing: 1px;
        box-shadow: 1px 1px 3px rgba(0,0,0,0.2);
    }
    .badge-native { background-color: #556B2F; }
    .badge-invasive { background-color: #A52A2A; } /* 磚紅色 */
    .badge-neutral { background-color: #DAA520; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# 詳細資料庫 (對應新 CSS 類別)
# ------------------------------------------------------
bird_info = {
    "家八哥": {
        "學名": "Acridotheres cristatellus formosanus",
        "中文名": "八哥（臺灣特有亞種）",
        "標籤": ["臺灣特有亞種", "珍貴稀有(II)", "近危(NT)"],
        "標籤顏色": ["native", "native", "native"],
        "外觀": """
        <div class="big-font">
            <ul>
                <li><b>整體特徵：</b> 雌雄同色，全身幾為純黑色。</li>
                <li><b>頭部關鍵：</b> 額羽聳立於喙基上如<b>冠羽</b>（這是最重要的辨識特徵）。虹膜橙黃色，喙象牙白色。</li>
                <li><b>翅膀與尾部：</b> 翼上有明顯白斑，初級覆羽先端和初級飛羽基部為白色，飛行時非常明顯。尾羽末端為白色，尾下覆羽黑白相間。</li>
                <li><b>腳部：</b> 跗蹠暗黃色。</li>
            </ul>
        </div>
        """,
        "習性": """
        <div class="big-font">
            <p><b>📍 棲地環境：</b><br>生活在海拔 2,100m 以下之竹林、疏林、開闊地區。常見於高速公路護欄、燈架、電線及農田牛背上。</p>
            <p><b>🐛 覓食策略：</b><br>雜食性。主要在地面覓食，常在耕地啄食蚯蚓、昆蟲、植物塊莖。也會在牛背上啄食體外寄生蟲。</p>
            <p><b>🥚 繁衍行為：</b><br>繁殖期 3-7 月，築巢於樹洞、電桿或鐵塔。一季可育兩窩，每窩產 3-5 枚卵（淡藍色或藍綠色）。具群聚性，清晨傍晚常聚大群。</p>
        </div>
        """,
        "保育": """
        <div class="big-font">
            <p><b>⚠️ 保育狀態：</b><br>台灣紅皮書列為<b>「近危 (NT)」</b>。野生動物保育法公告之<b>「珍貴稀有野生動物」</b>。</p>
            <p><b>⚔️ 生存威脅：</b><br>受到外來種八哥（家八哥、白尾八哥）的強勢競爭，巢位與食物資源被搶奪，導致野外數量快速減少。</p>
            <p><b>⚖️ 法規保護：</b><br>屬第二級珍貴稀有保育類，受法律保護。</p>
        </div>
        """
    },
    "common_myna": {
        "中文名": "家八哥",
        "學名": "Acridotheres tristis",
        "標籤": ["外來入侵種", "全球百大入侵種", "強勢物種"],
        "標籤顏色": ["invasive", "invasive", "invasive"],
        "外觀": """
        <div class="big-font">
            <ul>
                <li><b>整體特徵：</b> 全長約 25-26cm，身體褐色，頭及喉部黑色。</li>
                <li><b>頭部關鍵：</b> <b>眼周裸皮明顯呈黃色</b>（這是最明顯特徵），喙與腳呈亮黃色。無額前冠羽。</li>
                <li><b>翅膀與尾部：</b> 飛行時可見明顯的白色翼斑。尾羽黑色，末端白色。</li>
            </ul>
        </div>
        """,
        "習性": """
        <div class="big-font">
            <p><b>📍 棲地環境：</b><br>極度適應人類環境。遍布都市公園、校園、農地、垃圾場。</p>
            <p><b>🐛 覓食策略：</b><br>雜食性且機會主義者。昆蟲、果實、廚餘垃圾、小型脊椎動物皆吃。</p>
            <p><b>🥚 繁衍行為：</b><br>營穴巢，利用建築物縫隙、招牌、路標管洞築巢。繁殖力強，排擠原生鳥類。領域性強，噪鳴聲響亮且多變。</p>
        </div>
        """,
        "保育": """
        <div class="big-font">
            <p><b>⚠️ 入侵風險：</b><br>IUCN 全球百大入侵種之一。與原生八哥競爭巢位與食物，甚至捕食原生鳥類的蛋與雛鳥。</p>
            <p><b>📉 擴散機制：</b><br>早期因能模仿人語而被大量引入作為寵物，後逃逸或放生擴散。在台灣為強勢外來種，無保育等級。</p>
        </div>
        """
    },
    "javan_myna": {
        "中文名": "白尾八哥",
        "學名": "Acridotheres javanicus",
        "標籤": ["外來入侵種", "台灣數量最多", "原產地易危"],
        "標籤顏色": ["invasive", "invasive", "neutral"],
        "外觀": """
        <div class="big-font">
            <ul>
                <li><b>整體特徵：</b> 全身灰黑色為主，體型約 21-23cm。</li>
                <li><b>頭部關鍵：</b> 有短羽冠（不如原生八哥明顯），喙與腳為橘黃色。虹膜橘黃（幼鳥灰白）。</li>
                <li><b>尾部關鍵：</b> <b>尾羽末端及尾下覆羽為白色</b>，因此得名「白尾八哥」。</li>
                <li><b>辨識重點：</b> 與家八哥不同處在於全身偏灰黑且無眼周裸皮；與原生八哥不同處在於喙是黃色（原生為象牙白）且體色較灰。</li>
            </ul>
        </div>
        """,
        "習性": """
        <div class="big-font">
            <p><b>📍 棲地環境：</b><br>平原、近郊丘陵、都市草地。目前是台灣數量最多的外來八哥。</p>
            <p><b>🐛 覓食策略：</b><br>雜食性。喜愛在剛割完草的草地覓食昆蟲，也會吃人類垃圾。</p>
            <p><b>🥚 繁衍行為：</b><br>適應力極強，利用都市建築縫隙築巢。性情兇悍，常驅趕麻雀或其他鳥類。夜間有集體夜棲習性。</p>
        </div>
        """,
        "保育": """
        <div class="big-font">
            <p><b>⚠️ 入侵風險：</b><br>嚴重排擠原生八哥生存空間。在台灣野外已建立穩定且龐大的族群。</p>
            <p><b>📉 狀態描述：</b><br>國際自然保育聯盟 (IUCN) 在其原產地列為易危 (VU)，但在台灣是需要控制的入侵種。主要由籠鳥逃逸或宗教放生導致擴散。</p>
        </div>
        """
    }
}

LABEL_MAP = { "common_myna": "家八哥", "crested_myna": "八哥", "javan_myna": "白尾八哥" }
bird_info_clean = { "八哥": bird_info["家八哥"], "家八哥": bird_info["common_myna"], "白尾八哥": bird_info["javan_myna"] }

# ------------------------------------------------------
# 核心邏輯
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
        st.markdown("<div class='big-font' style='font-size:1.1rem !important;'>請上傳您拍攝到的八哥照片：</div>", unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"])
        
        st.markdown("---")
        st.markdown("### 辨識支援")
        st.markdown("""
        <div style=font-size: 1.1rem;">
        - 八哥 (原生種)<br>
        - 家八哥 (外來種)<br>
        - 白尾八哥 (外來種)
        </div>
        """, unsafe_allow_html=True)

    # --- 主畫面標題 ---
    st.markdown("<h1>🦜 野外八哥辨識圖鑑</h1>", unsafe_allow_html=True)
    st.markdown("<p class='big-font' style='color:#555;'>上傳照片，AI 將協助您辨識物種，並提供詳細的生態保育資訊。</p>", unsafe_allow_html=True)
    st.markdown("---")

    if not uploaded:
        st.warning("👈 請從左側側邊欄上傳圖片以開始分析")
        return

    # --- 分析結果區 ---
    col_img, col_stat = st.columns([0.8, 1.2], gap="large")

    image = Image.open(BytesIO(uploaded.read()))

    with col_img:
        st.markdown('<div class="bird-image-container">', unsafe_allow_html=True)
        st.image(image, caption="您的觀察照片", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_stat:
        with st.spinner("正在分析物種特徵..."):
            results = predict_image(model, labels, image)
        
        top_bird, top_prob = results[0]
        
        # 結果卡片
        st.markdown(f"""
        <div style="background-color: white; padding: 25px; border-radius: 12px; border-left: 8px solid #2F4F4F; box-shadow: 0 4px 10px rgba(0,0,0,0.1);">
            <h2 style="margin:0; border:none; padding:0; font-size: 2.8rem !important;">{top_bird}</h2>
            <p class="big-font" style="margin-top:10px; color: #556B2F;">信心指數：<b>{top_prob*100:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        # 標籤
        info = bird_info_clean.get(top_bird)
        if info:
            st.markdown("<div style='margin-top: 20px;'>", unsafe_allow_html=True)
            for t, c in zip(info.get("標籤", []), info.get("標籤顏色", [])):
                st.markdown(f"<span class='badge badge-{c}'>{t}</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown(f"<p class='big-font' style='margin-top:15px;'><b>學名：</b> <i>{info.get('學名')}</i></p>", unsafe_allow_html=True)

        # 圖表
        df = pd.DataFrame(results, columns=["物種", "機率"])
        df["機率(%)"] = (df["機率"] * 100).round(1)
        
        # Altair 字體調整
        base = alt.Chart(df).encode(
            y=alt.Y('物種', sort='-x', title=None, axis=alt.Axis(labelFontSize=18))
        )
        bar = base.mark_bar(color="#8FBC8F").encode(x=alt.X('機率(%)', title=None))
        text = base.mark_text(align='left', dx=5, font="Times New Roman", fontSize=14).encode(x='機率(%)', text='機率(%)')
        
        st.altair_chart((bar + text).properties(height=130), use_container_width=True)

    # --- 詳細資訊區 ---
    st.markdown("<h3 style='margin-top:30px;'>📖 物種詳細檔案</h3>", unsafe_allow_html=True)
    
    if info:
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
