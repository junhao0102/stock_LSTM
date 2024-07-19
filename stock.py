import requests
import streamlit as st
import pandas as pd 
from bs4 import BeautifulSoup

#從鉅亨網擷取特定股票（代碼8096）的歷史價格數據
# 發送 HTTP 請求獲取網頁內容，然後使用 BeautifulSoup 解析 HTML 並選擇表格行
url = "https://www.cnyes.com/archive/twstock/ps_historyprice/3231.htm"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
rows = soup.select("table tr")
company = soup.select("h1")[0].text.split()[0]

# 解析 HTML 表格數據並轉換為 pandas DataFrame
stock_data = []
for row in rows: 
    cols = row.find_all('td')
    #將空白的列排除
    if cols:
        row_data = [col.text for col in cols]
        stock_data.append(row_data)
stock_df = pd.DataFrame(stock_data, columns=['日期', '開盤價', '最高價', '最低價', '收盤價', '漲跌', '漲跌幅', '成交量', '成交金額', '未知'])

st.title(company)
st.write(stock_df)
