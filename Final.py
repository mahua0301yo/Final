# 載入必要模組
import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
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

# 定義繪圖函數來顯示股票數據和指標
def plot_stock_data(stock, strategy_name):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=(f"{strategy_name} 標的圖", "成交量"))

    fig.add_trace(go.Candlestick(x=stock['Date'], open=stock['Open'], high=stock['High'],
                                 low=stock['Low'], close=stock['Close'], name='K線圖'), row=1, col=1)
    
    # 添加指標數據
    if strategy_name == "Bollinger Bands":
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Middle_Band'], mode='lines', name='中軌'), row=1, col=1)
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Upper_Band'], mode='lines', name='上軌'), row=1, col=1)
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Lower_Band'], mode='lines', name='下軌'), row=1, col=1)
    elif strategy_name == "KDJ":
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['K'], mode='lines', name='K'), row=2, col=1)
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['D'], mode='lines', name='D'), row=2, col=1)
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['J'], mode='lines', name='J'), row=2, col=1)
    elif strategy_name == "RSI":
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['RSI'], mode='lines', name='RSI'), row=2, col=1)
    elif strategy_name == "MACD":
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['MACD'], mode='lines', name='MACD'), row=2, col=1)
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Signal_Line'], mode='lines', name='Signal Line'), row=2, col=1)
        fig.add_trace(go.Bar(x=stock['Date'], y=stock['Histogram'], name='Histogram'), row=2, col=1)
    elif strategy_name == "唐奇安通道":
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Donchian_High'], mode='lines', name='唐奇安上軌'), row=1, col=1)
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Donchian_Low'], mode='lines', name='唐奇安下軌'), row=1, col=1)
    
    # 添加成交量圖
    fig.add_trace(go.Bar(x=stock['Date'], y=stock['Volume'], name='成交量'), row=2, col=1)

    fig.update_layout(title=f"{strategy_name} 策略圖", xaxis_title='日期', yaxis_title='價格')
    st.plotly_chart(fig)

# 定義交易策略和績效計算函數
def trading_strategy(stock, strategy_name):
    # 這裡可以加入每個指標的交易策略，例如：
    if strategy_name == "Bollinger Bands":
        # 示例：布林通道策略
        stock['Buy_Signal'] = (stock['Close'] < stock['Lower_Band'])
        stock['Sell_Signal'] = (stock['Close'] > stock['Upper_Band'])
    elif strategy_name == "KDJ":
        # 示例：KDJ策略
        stock['Buy_Signal'] = (stock['K'] > stock['D']) & (stock['J'] < 20)
        stock['Sell_Signal'] = (stock['K'] < stock['D']) & (stock['J'] > 80)
    elif strategy_name == "RSI":
        # 示例：RSI策略
        stock['Buy_Signal'] = stock['RSI'] < 30
        stock['Sell_Signal'] = stock['RSI'] > 70
    elif strategy_name == "MACD":
        # 示例：MACD策略
        stock['Buy_Signal'] = stock['MACD'] > stock['Signal_Line']
        stock['Sell_Signal'] = stock['MACD'] < stock['Signal_Line']
    elif strategy_name == "唐奇安通道":
        # 示例：唐奇安通道策略
        stock['Buy_Signal'] = stock['Close'] > stock['Donchian_High'].shift(1)
        stock['Sell_Signal'] = stock['Close'] < stock['Donchian_Low'].shift(1)
    
    # 績效計算
    stock['Position'] = 0
    stock.loc[stock['Buy_Signal'], 'Position'] = 1
    stock.loc[stock['Sell_Signal'], 'Position'] = -1
    stock['Daily_Return'] = stock['Close'].pct_change()
    stock['Strategy_Return'] = stock['Position'].shift(1) * stock['Daily_Return']
    stock['Cumulative_Strategy_Return'] = (1 + stock['Strategy_Return']).cumprod() - 1

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
        elif strategy_name == "KDJ":
            stock = calculate_kdj(stock, period=kdj_period)
        elif strategy_name == "RSI":
            stock = calculate_rsi(stock, period=rsi_period)
        elif strategy_name == "MACD":
            stock = calculate_macd(stock, short_window=short_window, long_window=long_window, signal_window=signal_window)
        elif strategy_name == "唐奇安通道":
            stock = calculate_donchian_channels(stock, period=donchian_period)
        
        stock = trading_strategy(stock, strategy_name)
        plot_stock_data(stock, strategy_name)
        st.write(f"策略績效 (累積報酬): {stock['Cumulative_Strategy_Return'].iloc[-1]:.2%}")

if __name__ == "__main__":
    main()
