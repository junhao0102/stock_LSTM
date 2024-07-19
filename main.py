import twstock
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

# 設定目標股票代碼並抓取該股票的歷史價格數據
target_stock = '3231'
stock = twstock.Stock(target_stock)
target_price = stock.fetch_from(2020, 5)

# 定義數據框架的列名並創建數據框架
name_attribute = ['Date','Capacity', 'Turnover', 'Open', 'High', 'Low', 'Close', 'Change', 'Transaction']
df = pd.DataFrame(data=target_price, columns=name_attribute)
stock_df = df[['Date', 'Open']]

# 顯示股票代碼和數據框架
st.title(f"股票代碼: {target_stock}")

# 設置圖表樣式並繪製股票開盤價趨勢圖
plt.style.use('seaborn-darkgrid')
plt.xlabel('Date')
plt.ylabel('Price')
plt.plot(stock_df['Date'], stock_df['Open'])
st.pyplot(plt)
plt.clf()  

# 初始化縮放器並將開盤價數據縮放到0和1之間
scaler = MinMaxScaler(feature_range=(0, 1))

#reshape(row,column) 將1D數組轉換為2D數組 
scaled_prices = scaler.fit_transform(stock_df['Open'].values.reshape(-1, 1))

# 設置移動窗口大小並創建特徵和標籤
moving_size = 60
all_x, all_y = [], []
for i in range(len(scaled_prices) - moving_size):
    all_x.append(scaled_prices[i:i + moving_size])
    all_y.append(scaled_prices[i + moving_size])

# 將 x 和 y 轉換為 numpy 數組
all_x, all_y = np.array(all_x), np.array(all_y)


#LSTM 模型的輸入為3D，形狀為 (samples, timesteps, features)
all_x = np.reshape(all_x, (all_x.shape[0], all_x.shape[1], 1))

# 分割數據集為訓練集和測試集
Data_split = 0.8
train_size = int(len(all_x) * Data_split)
train_x, test_x = all_x[:train_size], all_x[train_size:]
train_y, test_y = all_y[:train_size], all_y[train_size:]

# 構建 LSTM 模型
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(train_x.shape[1], 1)),
    LSTM(units=50, return_sequences=False),
    Dense(units=25),
    Dense(units=1)
])

# 編譯模型並設置損失函數和優化器
model.compile(optimizer="adam", loss="mean_squared_error")

# 設置早停回調函數
callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 訓練模型
model.fit(train_x, train_y, batch_size=10, epochs=100, validation_split=0.2, callbacks=[callback])

# 預測測試集數據
preds = model.predict(test_x)

# 將預測結果轉換為原始數據
preds = scaler.inverse_transform(preds)


# 創建訓練集和測試集的數據框架並添加預測結果
train_df = stock_df.iloc[:train_size + moving_size]
test_df = stock_df.iloc[train_size + moving_size:]
test_df['Predictions'] = preds

# 繪製訓練集和測試集的實際價格和預測價格
plt.xlabel('Date')
plt.ylabel('Price')
plt.plot(train_df['Date'], train_df['Open'], label='Train',linewidth=1)
plt.plot(test_df['Date'], test_df['Open'], label='Test', linewidth=1)
plt.plot(test_df['Date'], test_df['Predictions'], label='Predictions', linewidth=2)
plt.legend()
st.pyplot(plt)
plt.clf() 


# 繪製測試集的實際價格和預測價格
plt.xlabel('Date')
plt.ylabel('Price')
plt.plot(train_df['Date'][-20:], train_df['Open'][-20:], label='Train',linewidth=1)
plt.plot(test_df['Date'][:30], test_df['Open'][:30], label='Test', linewidth=1)
plt.plot(test_df['Date'][:30], test_df['Predictions'][:30], label='Predictions', linewidth=2)
plt.legend()
st.pyplot(plt)
plt.clf()  