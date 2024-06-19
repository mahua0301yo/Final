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

# 定義函數來計算RSI指標，並加入超買和超賣標記
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
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("布林通道", "成交量"))

    fig.add_trace(go.Candlestick(x=stock['Date'], open=stock['Open'], high=stock['High'],
                                 low=stock['Low'], close=stock['Close'], name='K線圖'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Middle_Band'], mode='lines', name='中軌'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Upper_Band'], mode='lines', name='上軌'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Lower_Band'], mode='lines', name='下軌'), row=1, col=1)
    fig.add_trace(go.Bar(x=stock['Date'], y=stock['Volume'], name='成交量'), row=2, col=1)

    fig.update_layout(title="布林通道策略圖", xaxis_title='日期', yaxis_title='價格', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

def plot_kdj(stock):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("KDJ指標", "KDJ值"))

    fig.add_trace(go.Candlestick(x=stock['Date'], open=stock['Open'], high=stock['High'],
                                 low=stock['Low'], close=stock['Close'], name='K線圖'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['K'], mode='lines', name='K值'), row=2, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['D'], mode='lines', name='D值'), row=2, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['J'], mode='lines', name='J值'), row=2, col=1)

    fig.update_layout(title="KDJ策略圖", xaxis_title='日期', yaxis_title='價格', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

def plot_rsi(stock):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("RSI指標", "RSI值"))

    fig.add_trace(go.Candlestick(x=stock['Date'], open=stock['Open'], high=stock['High'],
                                 low=stock['Low'], close=stock['Close'], name='K線圖'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['RSI'], mode='lines', name='RSI'), row=2, col=1)
    
    # 加入超買和超賣的標記
    fig.add_trace(go.Scatter(x=stock['Date'], y=np.where(stock['Overbought'], 80, None),
                             mode='markers', marker=dict(color='red', size=10), name='超買 >80'), row=2, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=np.where(stock['Oversold'], 20, None),
                             mode='markers', marker=dict(color='blue', size=10), name='超賣 <20'), row=2, col=1)

    fig.update_layout(title="RSI策略圖", xaxis_title='日期', yaxis_title='價格', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)


def plot_macd(stock):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("MACD指標", "MACD值"))

    fig.add_trace(go.Candlestick(x=stock['Date'], open=stock['Open'], high=stock['High'],
                                 low=stock['Low'], close=stock['Close'], name='K線圖'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['MACD'], mode='lines', name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Signal_Line'], mode='lines', name='Signal Line'), row=2, col=1)
    fig.add_trace(go.Bar(x=stock['Date'], y=stock['Histogram'], name='Histogram'), row=2, col=1)

    fig.update_layout(title="MACD策略圖", xaxis_title='日期', yaxis_title='價格', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

def plot_donchian_channels(stock):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("唐奇安通道", "成交量"))

    fig.add_trace(go.Candlestick(x=stock['Date'], open=stock['Open'], high=stock['High'],
                                 low=stock['Low'], close=stock['Close'], name='K線圖'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Donchian_High'], mode='lines', name='唐奇安通道上界'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Donchian_Low'], mode='lines', name='唐奇安通道下界'), row=1, col=1)
    fig.add_trace(go.Bar(x=stock['Date'], y=stock['Volume'], name='成交量'), row=2, col=1)

    fig.update_layout(title="唐奇安通道策略圖", xaxis_title='日期', yaxis_title='價格', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

# 定義交易策略函數，加入止損和移動止損
def trading_strategy(stock, strategy_name, stop_loss_pct=5, trailing_stop_pct=3):
    stock['Position'] = np.nan

    # 根據不同的策略進行交易
    if strategy_name == "Bollinger Bands":
        stock = calculate_bollinger_bands(stock)
        stock['Position'] = np.where(stock['Close'] > stock['Upper_Band'], -1, np.nan)
        stock['Position'] = np.where(stock['Close'] < stock['Lower_Band'], 1, stock['Position'])
    
    elif strategy_name == "KDJ":
        stock = calculate_kdj(stock)
        stock['Position'] = np.where(stock['K'] > stock['D'], 1, -1)
    
    elif strategy_name == "RSI":
        stock = calculate_rsi(stock)
        stock['Position'] = np.where(stock['RSI'] < 20, 1, np.nan)
    
    elif strategy_name == "MACD":
        stock = calculate_macd(stock)
        stock['Position'] = np.where(stock['MACD'] > stock['Signal_Line'], 1, -1)
    
    elif strategy_name == "Donchian Channels":
        stock = calculate_donchian_channels(stock)
        stock['Position'] = np.where(stock['Close'] > stock['Donchian_High'], -1, np.nan)
        stock['Position'] = np.where(stock['Close'] < stock['Donchian_Low'], 1, stock['Position'])

    # 添加止損和移動止損功能
    stock['Stop_Loss'] = np.nan
    stock['Trailing_Stop'] = np.nan
    entry_price = None
    stop_loss_price = None

    for i in range(1, len(stock)):
        if not np.isnan(stock['Position'].iloc[i]):
            if stock['Position'].iloc[i] == 1:  # 如果持有多頭頭寸
                if entry_price is None:
                    entry_price = stock['Close'].iloc[i]
                    stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
                else:
                    # 更新移動止損水平
                    trailing_stop_price = entry_price * (1 + trailing_stop_pct / 100)
                    stop_loss_price = max(stop_loss_price, trailing_stop_price)
                
                stock['Stop_Loss'].iloc[i] = stop_loss_price

                # 檢查是否觸發止損
                if stock['Low'].iloc[i] <= stop_loss_price:
                    stock['Position'].iloc[i] = np.nan
                    entry_price = None
                    stop_loss_price = None

            elif stock['Position'].iloc[i] == -1:  # 如果持有空頭頭寸
                if entry_price is None:
                    entry_price = stock['Close'].iloc[i]
                    stop_loss_price = entry_price * (1 + stop_loss_pct / 100)
                else:
                    # 更新移動止損水平
                    trailing_stop_price = entry_price * (1 - trailing_stop_pct / 100)
                    stop_loss_price = min(stop_loss_price, trailing_stop_price)
                
                stock['Stop_Loss'].iloc[i] = stop_loss_price

                # 檢查是否觸發止損
                if stock['High'].iloc[i] >= stop_loss_price:
                    stock['Position'].iloc[i] = np.nan
                    entry_price = None
                    stop_loss_price = None

    return stock

# 主程式
def main():
    st.title('股票交易策略模擬')

    # 輸入股票代碼和日期範圍
    stockname = st.sidebar.text_input("請輸入股票代碼 (例如: AAPL, GOOGL):", value='AAPL')
    start_date = st.sidebar.date_input("請選擇開始日期:")
    end_date = st.sidebar.date_input("請選擇結束日期:")
    strategy_name = st.sidebar.selectbox('請選擇交易策略:', ['Bollinger Bands', 'KDJ', 'RSI', 'MACD', 'Donchian Channels'])

    if start_date >= end_date:
        st.sidebar.error('結束日期必須大於開始日期')
        return

    # 載入股票數據
    interval = '1d'  # 日線數據
    stock = load_stock_data(stockname, start_date, end_date, interval)

    if stock is not None:
        # 執行選定的交易策略
        stock = trading_strategy(stock, strategy_name)

        # 繪製策略圖表
        if strategy_name == 'Bollinger Bands':
            plot_bollinger_bands(stock)
        elif strategy_name == 'KDJ':
            plot_kdj(stock)
        elif strategy_name == 'RSI':
            plot_rsi(stock)
        elif strategy_name == 'MACD':
            plot_macd(stock)
        elif strategy_name == 'Donchian Channels':
            plot_donchian_channels(stock)

        # 顯示股票數據和交易信號
        st.subheader('股票數據和交易信號:')
        st.write(stock[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Position', 'Stop_Loss', 'Trailing_Stop']])
    
if __name__ == '__main__':
    main()
