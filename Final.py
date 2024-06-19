import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import datetime

# 函數來讀取股票數據
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

# 函數來計算MACD指標
def calculate_macd(stock, short_window=12, long_window=26, signal_window=9):
    stock['EMA_short'] = stock['Close'].ewm(span=short_window, adjust=False).mean()
    stock['EMA_long'] = stock['Close'].ewm(span=long_window, adjust=False).mean()
    stock['MACD'] = stock['EMA_short'] - stock['EMA_long']
    stock['Signal_Line'] = stock['MACD'].ewm(span=signal_window, adjust=False).mean()
    stock['Histogram'] = stock['MACD'] - stock['Signal_Line']
    return stock

# 函數來計算RSI指標
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

# 函數來計算KDJ指標
def calculate_kdj(stock, period=14):
    low_min = stock['Low'].rolling(window=period).min()
    high_max = stock['High'].rolling(window=period).max()
    rsv = (stock['Close'] - low_min) / (high_max - low_min) * 100
    stock['K'] = rsv.ewm(com=2).mean()
    stock['D'] = stock['K'].ewm(com=2).mean()
    stock['J'] = 3 * stock['K'] - 2 * stock['D']
    return stock

# 函數來繪製MACD圖
def plot_macd(stock):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("MACD指標", "MACD值"))

    fig.add_trace(go.Candlestick(x=stock['Date'], open=stock['Open'], high=stock['High'],
                                 low=stock['Low'], close=stock['Close'], name='K線圖'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['MACD'], mode='lines', name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Signal_Line'], mode='lines', name='Signal Line'), row=2, col=1)
    fig.add_trace(go.Bar(x=stock['Date'], y=stock['Histogram'], name='Histogram'), row=2, col=1)

    fig.update_layout(title="MACD策略圖", xaxis_title='日期', yaxis_title='價格')
    return fig

# 函數來繪製RSI圖
def plot_rsi(stock):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("RSI指標", "RSI值"))

    fig.add_trace(go.Candlestick(x=stock['Date'], open=stock['Open'], high=stock['High'],
                                 low=stock['Low'], close=stock['Close'], name='K線圖'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['RSI'], mode='lines', name='RSI'), row=2, col=1)

    fig.update_layout(title="RSI策略圖", xaxis_title='日期', yaxis_title='價格')
    return fig

# 函數來繪製KDJ圖
def plot_kdj(stock):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("KDJ指標", "KDJ值"))

    fig.add_trace(go.Candlestick(x=stock['Date'], open=stock['Open'], high=stock['High'],
                                 low=stock['Low'], close=stock['Close'], name='K線圖'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['K'], mode='lines', name='K值'), row=2, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['D'], mode='lines', name='D值'), row=2, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['J'], mode='lines', name='J值'), row=2, col=1)

    fig.update_layout(title="KDJ策略圖", xaxis_title='日期', yaxis_title='價格')
    return fig

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
    strategy_name = st.sidebar.selectbox("選擇指標", ["MACD", "RSI", "KDJ"])

    if strategy_name == "MACD":
        short_window = st.sidebar.slider("短期EMA窗口", min_value=5, max_value=50, value=12, step=1)
        long_window = st.sidebar.slider("長期EMA窗口", min_value=10, max_value=100, value=26, step=1)
        signal_window = st.sidebar.slider("信號線窗口", min_value=5, max_value=50, value=9, step=1)
    elif strategy_name == "RSI":
        rsi_period = st.sidebar.slider("RSI週期", min_value=5, max_value=50, value=14, step=1)
    elif strategy_name == "KDJ":
        kdj_period = st.sidebar.slider("KDJ週期", min_value=5, max_value=50, value=14, step=1)

    stock = load_stock_data(stockname, start_date, end_date, interval)
    if stock is not None:
        st.subheader(f"股票代號: {stockname}")
        st.write(stock.head())

        if strategy_name == "MACD":
            stock = calculate_macd(stock, short_window=short_window, long_window=long_window, signal_window=signal_window)
            fig_macd = plot_macd(stock)
            st.plotly_chart(fig_macd)
        elif strategy_name == "RSI":
            stock = calculate_rsi(stock, period=rsi_period)
            fig_rsi = plot_rsi(stock)
            st.plotly_chart(fig_rsi)
        elif strategy_name == "KDJ":
            stock = calculate_kdj(stock, period=kdj_period)
            fig_kdj = plot_kdj(stock)
            st.plotly_chart(fig_kdj)

if __name__ == "__main__":
    main()
