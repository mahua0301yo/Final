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
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 8))

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
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 8))

    # 上面的圖：價格走勢和KDJ指標
    ax1.plot(stock['Date'], stock['Close'], label='Close Price', color='blue')
    ax1.set_title('KDJ Indicator')
    ax1.set_ylabel('Price')
    ax1.legend()

    ax2.plot(stock['Date'], stock['K'], label='K', color='red')
    ax2.plot(stock['Date'], stock['D'], label='D', color='green')
    ax2.plot(stock['Date'], stock['J'], label='J', color='blue')
    ax2.fill_between(stock['Date'], 0, 20, where=stock['Overbought'], facecolor='red', alpha=0.3)
    ax2.fill_between(stock['Date'], 80, 100, where=stock['Oversold'], facecolor='green', alpha=0.3)
    ax2.set_title('KDJ Values')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('KDJ')
    ax2.legend()

    plt.tight_layout()
    st.pyplot(fig)

def plot_rsi(stock):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 8))

    # 上面的圖：價格走勢和RSI指標
    ax1.plot(stock['Date'], stock['Close'], label='Close Price', color='blue')
    ax1.set_title('RSI Indicator')
    ax1.set_ylabel('Price')
    ax1.legend()

    ax2.plot(stock['Date'], stock['RSI'], label='RSI', color='purple')
    ax2.fill_between(stock['Date'], 0, 20, where=stock['Overbought'], facecolor='red', alpha=0.3)
    ax2.fill_between(stock['Date'], 80, 100, where=stock['Oversold'], facecolor='green', alpha=0.3)
    ax2.set_title('RSI Values')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('RSI')
    ax2.legend()

    plt.tight_layout()
    st.pyplot(fig)

def plot_macd(stock):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 8))

    # 上面的圖：價格走勢和MACD指標
    ax1.plot(stock['Date'], stock['Close'], label='Close Price', color='blue')
    ax1.set_title('MACD Indicator')
    ax1.set_ylabel('Price')
    ax1.legend()

    ax2.plot(stock['Date'], stock['MACD'], label='MACD', color='purple')
    ax2.plot(stock['Date'], stock['Signal_Line'], label='Signal Line', linestyle='--')
    ax2.bar(stock['Date'], stock['Histogram'], label='Histogram', color='gray')
    ax2.set_title('MACD Values')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('MACD')
    ax2.legend()

    plt.tight_layout()
    st.pyplot(fig)

def plot_donchian_channels(stock):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 8))

    # 上面的圖：價格走勢和唐奇安通道指標
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

# 定義交易策略
def trading_strategy(stock, strategy_name):
    # 根據不同的策略進行交易
    if strategy_name == "Bollinger Bands":
        stock['Position'] = np.where(stock['Close'] > stock['Upper_Band'], -1, np.nan)
        stock['Position'] = np.where(stock['Close'] < stock['Lower_Band'], 1, stock['Position'])
    elif strategy_name == "KDJ":
        stock['Position'] = np.where(stock['K'] > stock['D'], 1, -1)
    elif strategy_name == "RSI":
        stock['Position'] = np.where(stock['RSI'] < 20, 1, np.nan)
        stock['Position'] = np.where(stock['RSI'] > 80, -1, stock['Position'])
    elif strategy_name == "MACD":
        stock['Position'] = np.where(stock['MACD'] > stock['Signal_Line'], 1, -1)
    elif strategy_name == "唐奇安通道":
        stock['Position'] = np.where(stock['Close'] > stock['Donchian_High'].shift(1), 1, np.nan)
        stock['Position'] = np.where(stock['Close'] < stock['Donchian_Low'].shift(1), -1, stock['Position'])

    stock['Position'].fillna(method='ffill', inplace=True)
    stock['Position'].fillna(0, inplace=True)
    stock['Market_Return'] = stock['Close'].pct_change()
    stock['Strategy_Return'] = stock['Market_Return'] * stock['Position'].shift(1)
    stock['Cumulative_Strategy_Return'] = (1 + stock['Strategy_Return']).cumprod() - 1

    # 計算績效指標
    total_trades = len(stock[(stock['Position'] == 1) | (stock['Position'] == -1)])
    winning_trades = len(stock[(stock['Position'].shift(1) == 1) & (stock['Strategy_Return'] > 0)])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    stock['Drawdown'] = (stock['Cumulative_Strategy_Return'].cummax() - stock['Cumulative_Strategy_Return'])
    max_drawdown = stock['Drawdown'].max()

    total_profit = stock['Strategy_Return'].sum()
    consecutive_losses = (stock['Strategy_Return'] < 0).astype(int).groupby(stock['Strategy_Return'].ge(0).cumsum()).sum().max()

    st.write(f"策略績效指標:")
    st.write(f"勝率: {win_rate:.2%}")
    st.write(f"最大連續虧損: {consecutive_losses}")
    st.write(f"最大資金回落: {max_drawdown:.2%}")
    st.write(f"總損益: {total_profit:.2%}")

    return stock

# 主函數
def main():
    st.title("股票技術指標交易策略")

    # 選擇資料區間
    st.sidebar.subheader("選擇資料區間")
    start_date = st.sidebar.date_input('選擇開始日期', datetime.date(2020, 1, 1))
    end_date = st.sidebar.date_input('選擇結束日期', datetime.date(2023, 1, 1))
    stockname = st.sidebar.text_input('請輸入股票代號 (例: 2330.TW)', '2330.TW')

    # 選擇K線時間長
    interval_options = {"1天": "1d", "1星期": "1wk", "1個月": "1mo"}
    interval_label = st.sidebar.selectbox("選擇K線時間長", list(interval_options.keys()))
    interval = interval_options[interval_label]

    # 選擇指標和參數
    strategy_name = st.sidebar.selectbox("選擇指標", ["Bollinger Bands", "KDJ", "RSI", "MACD", "唐奇安通道"])

    if strategy_name == "Bollinger Bands":
        bollinger_period = st.sidebar.slider("布林通道週期", min_value=5, max_value=50, value=20, step=1)
        bollinger_std = st.sidebar.slider("布林通道標準差倍數", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    elif strategy_name == "KDJ":
        kdj_period = st.sidebar.slider("KDJ週期", min_value=5, max_value=50, value=14, step=1)
    elif strategy_name == "RSI":
        rsi_period = st.sidebar.slider("RSI週期", min_value=5, max_value=50, value=14, step=1)
    elif strategy_name == "MACD":
        short_window = st.sidebar.slider("短期EMA窗口", min_value=5, max_value=50, value=12, step=1)
        long_window = st.sidebar.slider("長期EMA窗口", min_value=10, max_value=100, value=26, step=1)
        signal_window = st.sidebar.slider("信號線窗口", min_value=5, max_value=50, value=9, step=1)
    elif strategy_name == "唐奇安通道":
        donchian_period = st.sidebar.slider("唐奇安通道週期", min_value=5, max_value=50, value=20, step=1)

    stock = load_stock_data(stockname, start_date, end_date, interval)
    if stock is not None:
        st.subheader(f"股票代號: {stockname}")
        st.write(stock.head())

        if strategy_name == "Bollinger Bands":
            stock = calculate_bollinger_bands(stock, period=bollinger_period, std_dev=bollinger_std)
            plot_bollinger_bands(stock)
        elif strategy_name == "KDJ":
            stock = calculate_kdj(stock, period=kdj_period)
            plot_kdj(stock)
        elif strategy_name == "RSI":
            stock = calculate_rsi(stock, period=rsi_period)
            plot_rsi(stock)
        elif strategy_name == "MACD":
            stock = calculate_macd(stock, short_window=short_window, long_window=long_window, signal_window=signal_window)
            plot_macd(stock)
        elif strategy_name == "唐奇安通道":
            stock = calculate_donchian_channels(stock, period=donchian_period)
            plot_donchian_channels(stock)
        
        stock = trading_strategy(stock, strategy_name)
        st.write(f"策略績效 (累積報酬): {stock['Cumulative_Strategy_Return'].iloc[-1]:.2%}")

if __name__ == "__main__":
    main()
