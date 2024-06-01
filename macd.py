# 載入必要模組
import os
import numpy as np
import datetime
import pandas as pd
import streamlit as st
import streamlit.components.v1 as stc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from talib import MACD

###### (1) 開始設定 ######
html_temp = """
		<div style="background-color:#3872fb;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">金融資料視覺化呈現 (金融看板) </h1>
		<h2 style="color:white;text-align:center;">Financial Dashboard </h2>
		</div>
		"""
stc.html(html_temp)

## 读取Pickle文件
df_original = pd.read_pickle('kbars_2330_2022-01-01-2022-11-18.pkl')

df_original = df_original.drop('Unnamed: 0', axis=1)

##### 選擇資料區間
st.subheader("選擇開始與結束的日期, 區間:2022-01-03 至 2022-11-18")
start_date = st.text_input('選擇開始日期 (日期格式: 2022-01-03)', '2022-01-03')
end_date = st.text_input('選擇結束日期 (日期格式: 2022-11-18)', '2022-11-18')
start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
df = df_original[(df_original['time'] >= start_date) & (df_original['time'] <= end_date)]

###### (2) 轉化為字典 ######:
KBar_dic = df.to_dict()
KBar_open_list = list(KBar_dic['open'].values())
KBar_dic['open'] = np.array(KBar_open_list)
KBar_dic['product'] = np.repeat('tsmc', KBar_dic['open'].size)
KBar_time_list = list(KBar_dic['time'].values())
KBar_time_list = [i.to_pydatetime() for i in KBar_time_list]  # Timestamp to datetime
KBar_dic['time'] = np.array(KBar_time_list)
KBar_low_list = list(KBar_dic['low'].values())
KBar_dic['low'] = np.array(KBar_low_list)
KBar_high_list = list(KBar_dic['high'].values())
KBar_dic['high'] = np.array(KBar_high_list)
KBar_close_list = list(KBar_dic['close'].values())
KBar_dic['close'] = np.array(KBar_close_list)
KBar_volume_list = list(KBar_dic['volume'].values())
KBar_dic['volume'] = np.array(KBar_volume_list)
KBar_amount_list = list(KBar_dic['amount'].values())
KBar_dic['amount'] = np.array(KBar_amount_list)

###### (3) 改變 KBar 時間長度 ######
Date = start_date.strftime("%Y-%m-%d")
st.subheader("設定一根 K 棒的時間長度(分鐘)")
cycle_duration = st.number_input('輸入一根 K 棒的時間長度(單位:分鐘, 一日=1440分鐘)', key="KBar_duration", value=1440)
cycle_duration = int(cycle_duration)
KBar = indicator_forKBar_short.KBar(Date, cycle_duration)

for i in range(KBar_dic['time'].size):
    time = KBar_dic['time'][i]
    open_price = KBar_dic['open'][i]
    close_price = KBar_dic['close'][i]
    low_price = KBar_dic['low'][i]
    high_price = KBar_dic['high'][i]
    qty = KBar_dic['volume'][i]
    tag = KBar.AddPrice(time, open_price, close_price, low_price, high_price, qty)

KBar_dic = {}
KBar_dic['time'] = KBar.TAKBar['time']
KBar_dic['product'] = np.repeat('tsmc', KBar_dic['time'].size)
KBar_dic['open'] = KBar.TAKBar['open']
KBar_dic['high'] = KBar.TAKBar['high']
KBar_dic['low'] = KBar.TAKBar['low']
KBar_dic['close'] = KBar.TAKBar['close']
KBar_dic['volume'] = KBar.TAKBar['volume']

###### (4) 計算各種技術指標 ######
##### 將K線 Dictionary 轉換成 Dataframe
KBar_df = pd.DataFrame(KBar_dic)

#####  (i) 移動平均線策略   #####
####  設定長短移動平均線的 K棒 長度:
st.subheader("設定計算長移動平均線(MA)的 K 棒數目(整數, 例如 10)")
LongMAPeriod = st.slider('選擇一個整數', 0, 100, 10)
st.subheader("設定計算短移動平均線(MA)的 K 棒數目(整數, 例如 2)")
ShortMAPeriod = st.slider('選擇一個整數', 0, 100, 2)

#### 計算長短移動平均線
KBar_df['MA_long'] = KBar_df['close'].rolling(window=LongMAPeriod).mean()
KBar_df['MA_short'] = KBar_df['close'].rolling(window=ShortMAPeriod).mean()

#### 尋找最後 NAN值的位置
last_nan_index_MA = KBar_df['MA_long'][::-1].index[KBar_df['MA_long'][::-1].apply(pd.isna)][0]

#####  (ii) RSI 策略   #####
#### 順勢策略
### 設定長短 RSI 的 K棒 長度:
st.subheader("設定計算長RSI的 K 棒數目(整數, 例如 10)")
LongRSIPeriod = st.slider('選擇一個整數', 0, 1000, 10)
st.subheader("設定計算短RSI的 K 棒數目(整數, 例如 2)")
ShortRSIPeriod = st.slider('選擇一個整數', 0, 1000, 2)

### 計算 RSI指標長短線, 以及定義中線
def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

KBar_df['RSI_long'] = calculate_rsi(KBar_df, LongRSIPeriod)
KBar_df['RSI_short'] = calculate_rsi(KBar_df, ShortRSIPeriod)
KBar_df['RSI_Middle'] = np.array([50] * len(KBar_dic['time']))

### 尋找最後 NAN值的位置
last_nan_index_RSI = KBar_df['RSI_long'][::-1].index[KBar_df['RSI_long'][::-1].apply(pd.isna)][0]

#####  (iii) MACD 策略   #####
st.subheader("設定MACD參數")
fastperiod = st.slider('選擇快線週期', 1, 50, 12)
slowperiod = st.slider('選擇慢線週期', 1, 50, 26)
signalperiod = st.slider('選擇訊號線週期', 1, 50, 9)

KBar_df['MACD'], KBar_df['MACD_signal'], KBar_df['MACD_hist'] = MACD(KBar_df['close'], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)

### 尋找最後 NAN值的位置
last_nan_index_MACD = KBar_df['MACD'][::-1].index[KBar_df['MACD'][::-1].apply(pd.isna)][0]

###### (5) 將 Dataframe 欄位名稱轉換  ###### 
KBar_df.columns = [i[0].upper() + i[1:] for i in KBar_df.columns]

###### (6) 畫圖 ######
st.subheader("畫圖")

##### K線圖, 移動平均線 MA
with st.expander("K線圖, 移動平均線"):
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
    #### include candlestick with rangeselector
    fig1.add_trace(go.Candlestick(x=KBar_df['Time'],
                    open=KBar_df['Open'], high=KBar_df['High'],
                    low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
                   secondary_y=True)
    
    #### include a go.Bar trace for volumes
    fig1.add_trace(go.Bar(x=KBar_df['Time'], y=KBar_df['Volume'], name='成交量', marker=dict(color='black', opacity=0.5)), secondary_y=False)
    
    #### 移動平均線
    fig1.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA:], y=KBar_df['Ma_long'][last_nan_index_MA:], mode='lines', name='MA_long'), secondary_y=True)
    fig1.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MA:], y=KBar_df['Ma_short'][last_nan_index_MA:], mode='lines', name='MA_short'), secondary_y=True)
    
    fig1.update_layout(title='K線圖和移動平均線', yaxis_title='價格')
    st.plotly_chart(fig1)

##### RSI 指標圖
with st.expander("RSI 指標圖"):
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_RSI:], y=KBar_df['Rsi_long'][last_nan_index_RSI:], mode='lines', name='RSI_long'))
    fig2.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_RSI:], y=KBar_df['Rsi_short'][last_nan_index_RSI:], mode='lines', name='RSI_short'))
    fig2.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_RSI:], y=KBar_df['Rsi_middle'][last_nan_index_RSI:], mode='lines', name='RSI_middle', line=dict(color='black', dash='dot')))
    
    fig2.update_layout(title='RSI 指標圖', yaxis_title='RSI 值', xaxis_title='時間')
    st.plotly_chart(fig2)

##### MACD 指標圖
with st.expander("MACD 指標圖"):
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MACD:], y=KBar_df['Macd'][last_nan_index_MACD:], mode='lines', name='MACD'))
    fig3.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MACD:], y=KBar_df['Macd_signal'][last_nan_index_MACD:], mode='lines', name='MACD_signal'))
    fig3.add_trace(go.Bar(x=KBar_df['Time'][last_nan_index_MACD:], y=KBar_df['Macd_hist'][last_nan_index_MACD:], name='MACD_hist'))
    
    fig3.update_layout(title='MACD 指標圖', yaxis_title='MACD 值', xaxis_title='時間')
    st.plotly_chart(fig3)
