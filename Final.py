# 載入必要模組
import yfinance as yf
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import datetime

# 定義函數來讀取股票數據
def load_stock_data(stockname, start_date, end_date, interval):
    stock = yf.download(stockname, start=start_date, end=end_date, interval=interval)
    if stock.empty:
        st.error("未能讀取到數據，請檢查股票代號是否正確")
        return None
    stock.rename(columns={'Volume': 'amount'}, inplace=True)
    stock.drop(columns=['Adj Close'], inplace=True)
    stock['Volume'] = (stock['amount'] / (stock['Open'] + stock['Close']) / 2).astype(int)
    stock.reset_index(inplace=True)
    return stock

# 定義函數來計算布林通道指標
def calculate_bollinger_bands(stock, period=20, std_dev=2):
    stock['Middle_Band'] = stock['Close'].rolling(window=period).mean()
    stock['Upper_Band'] = stock['Middle_Band'] + std_dev * stock['Close'].rolling(window=period).std()
    stock['Lower_Band'] = stock['Middle_Band'] - std_dev * stock['Close'].rolling(window=period).std()
    return stock

# 定義函數來計算KDJ指標
def calculate_kdj(stock, period=14):
    low_min = stock['Low'].rolling(window=period).min()
    high_max = stock['High'].rolling(window=period).max()
    rsv = (stock['Close'] - low_min) / (high_max - low_min) * 100
    stock['K'] = rsv.ewm(com=2).mean()
    stock['D'] = stock['K'].ewm(com=2).mean()
    stock['J'] = 3 * stock['K'] - 2 * stock['D']
    return stock

# 定義函數來計算RSI指標
def calculate_rsi(stock, period=14):
    delta = stock['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    stock['RSI'] = 100 - (100 / (1 + rs))
    
    # 新增超買和超賣標記
    stock['Overbought'] = stock['RSI'] > 70
    stock['Oversold'] = stock['RSI'] < 30
    
    return stock

# 定義函數來計算MACD指標
def calculate_macd(stock, short_window=12, long_window=26, signal_window=9):
    stock['EMA_short'] = stock['Close'].ewm(span=short_window, adjust=False).mean()
    stock['EMA_long'] = stock['Close'].ewm(span=long_window, adjust=False).mean()
    stock['MACD'] = stock['EMA_short'] - stock['EMA_long']
    stock['Signal_Line'] = stock['MACD'].ewm(span=signal_window, adjust=False).mean()
    stock['Histogram'] = stock['MACD'] - stock['Signal_Line']
    return stock

# 定義函數來計算唐奇安通道指標
def calculate_donchian_channels(stock, period=20):
    stock['Donchian_High'] = stock['High'].rolling(window=period).max()
    stock['Donchian_Low'] = stock['Low'].rolling(window=period).min()
    return stock

# 定義單獨的繪圖函數
def plot_bollinger_bands(stock):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # 上面的圖：價格走勢和布林通道
    ax1.plot(stock['Date'], stock['Close'], label='Close Price', color='blue')
    ax1.plot(stock['Date'], stock['Middle_Band'], label='Middle Band', linestyle='--')
    ax1.plot(stock['Date'], stock['Upper_Band'], label='Upper Band', linestyle='--')
    ax1.plot(stock['Date'], stock['Lower_Band'], label='Lower Band', linestyle='--')
    ax1.fill_between(stock['Date'], stock['Upper_Band'], stock['Lower_Band'], alpha=0.2, color='gray')
    ax1.set_title('Bollinger Bands')
    ax1.set_ylabel('Price')
    ax1.legend()
    
    # 下面的圖：成交量
    ax2.bar(stock['Date'], stock['Volume'], color='gray')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volume')
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_kdj(stock):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # 上面的圖：價格走勢
    ax1.plot(stock['Date'], stock['Close'], label='Close Price', color='blue')
    ax1.set_title('KDJ')
    ax1.set_ylabel('Price')
    
    # 下面的圖：KDJ指標
    ax2.plot(stock['Date'], stock['K'], label='K', linestyle='--')
    ax2.plot(stock['Date'], stock['D'], label='D', linestyle='--')
    ax2.plot(stock['Date'], stock['J'], label='J', linestyle='--')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('KDJ Value')
    ax2.legend()
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_rsi(stock):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # 上面的圖：價格走勢
    ax1.plot(stock['Date'], stock['Close'], label='Close Price', color='blue')
    ax1.set_title('RSI')
    ax1.set_ylabel('Price')
    
    # 下面的圖：RSI指標
    ax2.plot(stock['Date'], stock['RSI'], label='RSI', color='orange')
    ax2.axhline(y=70, color='r', linestyle='--')
    ax2.axhline(y=30, color='g', linestyle='--')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('RSI Value')
    ax2.legend(['RSI', 'Overbought', 'Oversold'])
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_macd(stock):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # 上面的圖：價格走勢
    ax1.plot(stock['Date'], stock['Close'], label='Close Price', color='blue')
    ax1.set_title('MACD')
    ax1.set_ylabel('Price')
    
    # 下面的圖：MACD指標和信號線
    ax2.plot(stock['Date'], stock['MACD'], label='MACD', color='red')
    ax2.plot(stock['Date'], stock['Signal_Line'], label='Signal Line', color='blue')
    ax2.bar(stock['Date'], stock['Histogram'], label='Histogram', color='gray')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('MACD')
    ax2.legend()
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_donchian_channels(stock):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # 上面的圖：價格走勢
    ax1.plot(stock['Date'], stock['Close'], label='Close Price', color='blue')
    ax1.plot(stock['Date'], stock['Donchian_High'], label='Donchian High', linestyle='--')
    ax1.plot(stock['Date'], stock['Donchian_Low'], label='Donchian Low', linestyle='--')
    ax1.fill_between(stock['Date'], stock['Donchian_High'], stock['Donchian_Low'], alpha=0.2, color='gray')
    ax1.set_title('Donchian Channels')
    ax1.set_ylabel('Price')
    ax1.legend()
    
    # 下面的圖：成交量
    ax2.bar(stock['Date'], stock['Volume'], color='gray')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volume')
    
    plt.tight_layout()
    st.pyplot(fig)

# Streamlit App 部分
def main():
    st.title('Technical Analysis App')
    
    # 使用者輸入股票代號和日期範圍
    stockname = st.text_input("請輸入股票代號 (例如 AAPL):")
    start_date = st.date_input("請選擇開始日期:")
    end_date = st.date_input("請選擇結束日期:")
    
    if st.button('執行'):
        # 讀取股票數據
        stock = load_stock_data(stockname, start_date, end_date, '1d')
        
        if stock is not None:
            # 計算技術指標
            stock = calculate_bollinger_bands(stock)
            stock = calculate_kdj(stock)
            stock = calculate_rsi(stock)
            stock = calculate_macd(stock)
            stock = calculate_donchian_channels(stock)
            
            # 顯示價格走勢圖和相關技術指標
            plot_bollinger_bands(stock)
            plot_kdj(stock)
            plot_rsi(stock)
            plot_macd(stock)
            plot_donchian_channels(stock)

# 執行應用
if __name__ == '__main__':
    main()
