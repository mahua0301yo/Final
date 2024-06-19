import yfinance as yf
import pandas as pd
import streamlit as st
import mplfinance as mpf
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
    stock['Overbought'] = stock['RSI'] > 80
    stock['Oversold'] = stock['RSI'] < 20
    
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
    # 繪製K線圖
    mpf.plot(stock.set_index('Date'), type='candle', volume=True, show_nontrading=True,
             title='布林通道策略圖', ylabel='價格', ylabel_lower='成交量', style='yahoo')

    # 繪製布林通道
    plt.plot(stock['Middle_Band'], label='中軌', color='blue')
    plt.plot(stock['Upper_Band'], label='上軌', color='red')
    plt.plot(stock['Lower_Band'], label='下軌', color='green')
    plt.legend()

    plt.show()

def plot_kdj(stock):
    # 繪製KDJ指標
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
    
    axes[0].set_title('KDJ策略圖')
    mpf.plot(stock.set_index('Date'), type='candle', ax=axes[0], volume=True, show_nontrading=True)
    
    axes[1].plot(stock['K'], label='K值', color='blue')
    axes[1].plot(stock['D'], label='D值', color='red')
    axes[1].plot(stock['J'], label='J值', color='green')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def plot_rsi(stock):
    # 繪製RSI指標
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
    
    axes[0].set_title('RSI策略圖')
    mpf.plot(stock.set_index('Date'), type='candle', ax=axes[0], volume=True, show_nontrading=True)
    
    axes[1].plot(stock['RSI'], label='RSI', color='blue')
    axes[1].axhline(y=80, color='r', linestyle='--')
    axes[1].axhline(y=20, color='g', linestyle='--')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def plot_macd(stock):
    # 繪製MACD指標
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
    
    axes[0].set_title('MACD策略圖')
    mpf.plot(stock.set_index('Date'), type='candle', ax=axes[0], volume=True, show_nontrading=True)
    
    axes[1].plot(stock['MACD'], label='MACD', color='blue')
    axes[1].plot(stock['Signal_Line'], label='Signal Line', color='red')
    axes[1].bar(stock.index, stock['Histogram'], color='gray', alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def plot_donchian_channels(stock):
    # 繪製唐奇安通道
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
    
    axes[0].set_title('唐奇安通道策略圖')
    mpf.plot(stock.set_index('Date'), type='candle', ax=axes[0], volume=True, show_nontrading=True)
    
    axes[0].plot(stock['Donchian_High'], label='高值通道', color='red')
    axes[0].plot(stock['Donchian_Low'], label='低值通道', color='green')
    axes[0].legend()

    plt.tight_layout()
    plt.show()

# 定義交易策略
def trading_strategy(stock, strategy_name):
    # 根據不同的策略進行交易
    if strategy_name == "Bollinger Bands":
        stock['Position'] = np.where(stock['Close'] > stock['Upper_Band'], -1, np.nan)
        stock['Position'] = np.where(stock['Close'] < stock['Lower_Band'], 1, stock['Position'])
    elif strategy_name == "KDJ":
        stock['Position'] = np.where(stock['K'] > stock['D'], 1, -1)
    elif strategy_name == "RSI":
        stock['Position'] = np.where(stock['RSI'] > 70, -1, np.nan)
        stock['Position'] = np.where(stock['RSI'] < 30, 1, stock['Position'])
    elif strategy_name == "MACD":
        stock['Position'] = np.where(stock['MACD'] > stock['Signal_Line'], 1, -1)
    elif strategy_name == "Donchian Channels":
        stock['Position'] = np.where(stock['Close'] > stock['Donchian_High'].shift(1), 1, np.nan)
        stock['Position'] = np.where(stock['Close'] < stock['Donchian_Low'].shift(1), -1, stock['Position'])
    
    return stock

# 主程式
def main():
    st.title('股票交易策略回測與可視化')
    
    # 使用 Streamlit 創建交互式界面
    st.sidebar.header('設置')
    stockname = st.sidebar.text_input('輸入股票代號（如AAPL）：', 'AAPL')
    start_date = st.sidebar.date_input('開始日期：', datetime.date(2021, 1, 1))
    end_date = st.sidebar.date_input('結束日期：', datetime.date(2021, 12, 31))
    interval = st.sidebar.selectbox('K線週期：', ['1d', '1wk', '1mo'], index=0)
    strategy_name = st.sidebar.selectbox('選擇交易策略：', ['Bollinger Bands', 'KDJ', 'RSI', 'MACD', 'Donchian Channels'], index=0)
    
    # 讀取股票數據
    stock = load_stock_data(stockname, start_date, end_date, interval)
    if stock is None:
        return
    
    # 計算技術指標
    if strategy_name == 'Bollinger Bands':
        stock = calculate_bollinger_bands(stock)
        plot_bollinger_bands(stock)
    elif strategy_name == 'KDJ':
        stock = calculate_kdj(stock)
        plot_kdj(stock)
    elif strategy_name == 'RSI':
        stock = calculate_rsi(stock)
        plot_rsi(stock)
    elif strategy_name == 'MACD':
        stock = calculate_macd(stock)
        plot_macd(stock)
    elif strategy_name == 'Donchian Channels':
        stock = calculate_donchian_channels(stock)
        plot_donchian_channels(stock)
    
    # 回測交易策略
    stock = trading_strategy(stock, strategy_name)
    
    # 顯示股票數據和交易信號
    st.subheader('股票數據和交易信號')
    st.write(stock)

if __name__ == '__main__':
    main()
