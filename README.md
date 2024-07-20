# 股票價格預測應用
使用LSTM神經網路模型來預測股票價格的Streamlit應用  

## 數據處理流程

1. 數據獲取：`twstock`庫獲取指定股票的歷史價格數據
2. 數據預處理：`MinMaxScaler`將價格數據縮放到0和1之間
3. 模型構建：`Keras`構建`LSTM`模型
4. 模型訓練：訓練`LSTM`模型
5. 預測：對測試數據集進行預測
6. 可視化：`Matplotlib`繪製實際價格和預測價格的對比圖

## 使用

安裝依賴的Python庫
```
pip install -r requirements.txt
```

啟動應用
```
streamlit run app.py
```


