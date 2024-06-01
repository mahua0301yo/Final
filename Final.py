# 載入必要模組
import os
# os.chdir(r'C:\Users\user\Dropbox\系務\專題實作\112\金融看板\for students')
#import haohaninfo
#from order_Lo8 import Record
import numpy as np
#from talib.abstract import SMA,EMA, WMA, RSI, BBANDS, MACD
#import sys
import indicator_f_Lo2_short,datetime, indicator_forKBar_short
import datetime
import pandas as pd
import streamlit as st 
import streamlit.components.v1 as stc 
#讀取股票
import yfinance as yf

###### (1) 開始設定 ######
html_temp = """
		<div style="background-color:#3872fb;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">金融資料視覺化呈現 (金融看板) </h1>
		<h2 style="color:white;text-align:center;">Financial Dashboard </h2>
		</div>
		"""
stc.html(html_temp)

# 驗證日期輸入
if start_date > end_date:
    st.error("開始日期不能晚於結束日期")
else:
    try:
        # 讀取股票資料
        stock = yf.download(stockname, start=start_date, end=end_date)
        
        if stock.empty:
            st.error("未能讀取到數據，請檢查股票代號是否正確")
        else:
            st.success("數據讀取成功")

            # 將 Volume 改為 amount
            stock.rename(columns={'Volume': 'amount'}, inplace=True)
            
            # 使用 amount / Open 計算新的 Volume 並轉換為整數
            stock['Volume'] = (stock['amount'] / stock['Open']).astype(int)
            
            # 互換 Volume 和 amount 兩欄的位置
            cols = stock.columns.tolist()
            vol_idx = cols.index('Volume')
            amt_idx = cols.index('amount')
            cols[vol_idx], cols[amt_idx] = cols[amt_idx], cols[vol_idx]
            stock = stock[cols]
            
            # 顯示股票數據
            st.write(stock)

            # 轉化為字典
            KBar_dic = stock.to_dict()
            KBar_dic['product'] = np.repeat(stockname, len(stock))
            
            # 進一步數據處理和可視化
            # ...
    
    except Exception as e:
        st.error(f"讀取數據時出錯: {e}")
