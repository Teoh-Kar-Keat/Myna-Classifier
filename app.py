# -------------------
# 右：預測結果卡片
# -------------------
with col2:
    st.markdown("<div class='right-card'>", unsafe_allow_html=True)
    if uploaded and image is not None:
        st.markdown("### 🔍 預測結果")

        results = predict_all(model, labels, image)
        results.sort(key=lambda x: x[1], reverse=True)

        # 卡片式機率顯示
        for i, (name, prob) in enumerate(results):
            color = "#32CD32" if i == 0 else "#87CEFA"
            st.markdown(f"""
            <div style='background-color:{color};
                        padding:12px; border-radius:15px;
                        margin-bottom:8px;
                        box-shadow:2px 2px 5px rgba(0,0,0,0.2);'>
                <h3 style='color:white; margin:0;'>{name}: {prob*100:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)

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

        # -------------------
        # 類別資訊區塊
        # -------------------
        st.markdown("### 📖 類別資訊")
        # 自訂每個類別描述
        info_map = {
            "家八哥": "中型鳥類，體羽黑亮帶白色翼斑，常見於城市與農村環境。",
            "八哥": "羽色光亮黑色，頭頂羽冠明顯，性格活潑好動。",
            "白尾八哥": "主要特徵為尾羽白色，喙黑色，喜群居生活。"
        }
        for name, _ in results:
            desc = info_map.get(name, "暫無資料")
            st.markdown(f"**{name}**: {desc}")

    st.markdown("</div>", unsafe_allow_html=True)
