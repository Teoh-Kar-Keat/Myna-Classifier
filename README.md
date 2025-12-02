# 八哥辨識實作摘要

本程式示範以遷移學習辨識台灣三種八哥的流程。原始方法自 GitHub 下載 myna 資料集並解壓，將三個類別影像讀入、調整為 256×256 並轉為數值陣列，經 `ResNet50V2` 的 `preprocess_input` 處理後，使用 pretrained `ResNet50V2`（`include_top=False`、`pooling='avg'`）作特徵擷取器並凍結權重，接上單一 `Dense(3, softmax)` 層進行訓練。由於資料量極少（23 張），僅以整批訓練示範，未切分驗證集，最後以模型 `predict` 生成分類結果，並透過 Gradio 建置簡易網頁介面。

後續進行介面優化，將 Gradio 改為 Streamlit 應用，增加標題、圖文分欄與機率可視化，提升使用者互動體驗；同時為每個偵測物種新增生態介紹，使辨識結果兼具教育與科普價值。整體而言，本筆記本結合遷移學習實作與介面優化，呈現一個清楚易懂且生態化的八哥辨識應用示範。

原github鏈接：  
https://github.com/yenlung/Deep-Learning-Basics/blob/master/colab02c%20%E7%94%A8%E9%81%B7%E7%A7%BB%E5%AD%B8%E7%BF%92%E6%89%93%E9%80%A0%E5%85%AB%E5%93%A5%E8%BE%A8%E8%AD%98AI.ipynb


agent 開發過程：  
https://github.com/Teoh-Kar-Keat/Myna-Classifier/blob/main/Myna%20bird%20Streamlit%20app.pdf


https://github.com/user-attachments/assets/9607bc61-b640-42f7-b7a6-38a08de7ead3

