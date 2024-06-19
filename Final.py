# 載入必要模組
import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots

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
    stock['EMA12'] = stock['Close'].ewm(span=short_window, adjust=False).mean()
    stock['EMA26'] = stock['Close'].ewm(span=long_window, adjust=False).mean()
    stock['MACD'] = stock['EMA12'] - stock['EMA26']
    stock['Signal_Line'] = stock['MACD'].ewm(span=signal_window, adjust=False).mean()
    stock.drop(columns=['EMA12', 'EMA26'], inplace=True)
    return stock

# 定義函數來計算唐奇安通道指標
def calculate_donchian_channels(stock, period=20):
    stock['Upper_Channel'] = stock['Close'].rolling(window=period).max()
    stock['Lower_Channel'] = stock['Close'].rolling(window=period).min()
    return stock

# 繪製股票數據和指標圖
def plot_stock_data(stock, strategy_name):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02, row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(x=stock['Date'],
                                 open=stock['Open'],
                                 high=stock['High'],
                                 low=stock['Low'],
                                 close=stock['Close'],
                                 name='價格'), row=1, col=1)

    if 'Upper_Band' in stock.columns:
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Upper_Band'], line=dict(color='red', width=1), name='上軌'), row=1, col=1)
    if 'Middle_Band' in stock.columns:
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Middle_Band'], line=dict(color='blue', width=1), name='中軌'), row=1, col=1)
    if 'Lower_Band' in stock.columns:
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Lower_Band'], line=dict(color='red', width=1), name='下軌'), row=1, col=1)

    # 加入唐奇安通道（使用實線）
    if 'Upper_Channel' in stock.columns:
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Upper_Channel'], line=dict(color='green', width=1), name='唐奇安通道上通道'), row=1, col=1)
    if 'Lower_Channel' in stock.columns:
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Lower_Channel'], line=dict(color='green', width=1), name='唐奇安通道下通道'), row=1, col=1)
    
    fig.add_trace(go.Bar(x=stock['Date'], y=stock['amount'], name='交易量'), row=2, col=1)

    fig.update_layout(title=f"{strategy_name}策略 - 股票價格與交易量",
                      xaxis_title='日期',
                      yaxis_title='價格')

    st.plotly_chart(fig)

# 繪製KDJ指標
def plot_kdj(stock):
    plot_stock_data(stock, "KDJ")

    fig_kdj = go.Figure()
    fig_kdj.add_trace(go.Scatter(x=stock['Date'], y=stock['K'], line=dict(color='blue', width=1), name='K'))
    fig_kdj.add_trace(go.Scatter(x=stock['Date'], y=stock['D'], line=dict(color='orange', width=1), name='D'))
    fig_kdj.add_trace(go.Scatter(x=stock['Date'], y=stock['J'], line=dict(color='green', width=1), name='J'))
    
    fig_kdj.update_layout(title='KDJ指標',
                          xaxis_title='日期',
                          yaxis_title='數值')
    st.plotly_chart(fig_kdj)

# 繪製RSI指標
def plot_rsi(stock):
    plot_stock_data(stock, "RSI")

    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=stock['Date'], y=stock['RSI'], line=dict(color='purple', width=1), name='RSI'))
    
    # 加入超買區和超賣區的背景色
    fig_rsi.add_shape(type="rect", xref="paper", yref="y",
                      x0=0, y0=80, x1=1, y1=100,
                      fillcolor="rgba(255, 0, 0, 0.3)",  # 紅色半透明填充
                      layer="below", line_width=0,
                      name="超買區域")
    
    fig_rsi.add_shape(type="rect", xref="paper", yref="y",
                      x0=0, y0=0, x1=1, y1=20,
                      fillcolor="rgba(0, 0, 255, 0.3)",  # 藍色半透明填充
                      layer="below", line_width=0,
                      name="超賣區域")
    
    fig_rsi.update_layout(title='RSI指標',
                          xaxis_title='日期',
                          yaxis_title='數值')
    st.plotly_chart(fig_rsi)


# 繪製MACD指標
def plot_macd(stock):
    plot_stock_data(stock, "MACD")

    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=stock['Date'], y=stock['MACD'], line=dict(color='blue', width=1), name='MACD'))
    fig_macd.add_trace(go.Scatter(x=stock['Date'], y=stock['Signal_Line'], line=dict(color='red', width=1), name='Signal Line'))

    fig_macd.add_trace(go.Bar(x=stock['Date'], y=stock['MACD'] - stock['Signal_Line'], marker_color='grey', name='MACD Histogram'))

    fig_macd.update_layout(title='MACD指標',
                           xaxis_title='日期',
                           yaxis_title='數值')
    st.plotly_chart(fig_macd)

# 策略設計和回測函數

# 1. 布林通道策略
def bollinger_band_strategy(stock, period=20, std_dev=2):
    stock = calculate_bollinger_bands(stock, period=period, std_dev=std_dev)
    
    # 產生交易訊號
    stock['Buy_Signal'] = (stock['Close'] > stock['Upper_Band']).astype(int)
    stock['Sell_Signal'] = (stock['Close'] < stock['Lower_Band']).astype(int)
    
    # 計算策略回報率和其他績效指標
    signals, win_rate = calculate_strategy_performance(stock, 'Buy_Signal', 'Sell_Signal')
    
    return signals, win_rate

# 2. KDJ策略
def kdj_strategy(stock, period=14):
    stock = calculate_kdj(stock, period=period)
    
    # 產生交易訊號
    stock['Buy_Signal'] = (stock['K'] > stock['D']).astype(int)
    stock['Sell_Signal'] = (stock['K'] < stock['D']).astype(int)
    
    # 計算策略回報率和其他績效指標
    signals, win_rate = calculate_strategy_performance(stock, 'Buy_Signal', 'Sell_Signal')
    
    return signals, win_rate

# 3. RSI策略
def rsi_strategy(stock, period=14):
    stock = calculate_rsi(stock, period=period)
    
    # 產生交易訊號
    stock['Buy_Signal'] = (stock['RSI'] < 30).astype(int)
    stock['Sell_Signal'] = (stock['RSI'] > 70).astype(int)
    
    # 計算策略回報率和其他績效指標
    signals, win_rate = calculate_strategy_performance(stock, 'Buy_Signal', 'Sell_Signal')
    
    return signals, win_rate

# 4. MACD策略
def macd_strategy(stock, short_window=12, long_window=26, signal_window=9):
    stock = calculate_macd(stock, short_window=short_window, long_window=long_window, signal_window=signal_window)
    
    # 產生交易訊號
    stock['Buy_Signal'] = (stock['MACD'] > stock['Signal_Line']).astype(int)
    stock['Sell_Signal'] = (stock['MACD'] < stock['Signal_Line']).astype(int)
    
    # 計算策略回報率和其他績效指標
    signals, win_rate = calculate_strategy_performance(stock, 'Buy_Signal', 'Sell_Signal')
    
    return signals, win_rate

# 5. 唐奇安通道策略
def donchian_channel_strategy(stock, period=20):
    stock = calculate_donchian_channels(stock, period=period)
    
    # 產生交易訊號
    stock['Buy_Signal'] = (stock['Close'] > stock['Upper_Channel']).astype(int)
    stock['Sell_Signal'] = (stock['Close'] < stock['Lower_Channel']).astype(int)
    
    # 計算策略回報率和其他績效指標
    signals, win_rate = calculate_strategy_performance(stock, 'Buy_Signal', 'Sell_Signal')
    
    return signals, win_rate

# 計算策略回測績效
def calculate_strategy_performance(stock, buy_signal_col, sell_signal_col):
    # 初始化交易訊號為0
    stock['signal'] = 0
    
    # 買入訊號為1，賣出訊號為-1
    stock.loc[stock[buy_signal_col] == 1, 'signal'] = 1
    stock.loc[stock[sell_signal_col] == 1, 'signal'] = -1
    
    # 計算每日收益率
    stock['Returns'] = stock['Close'].pct_change()
    stock['Strategy_Returns'] = stock['Returns'] * stock['signal'].shift(1)
    
    # 計算持有期間
    stock['Trade_Duration'] = stock.index.to_series().diff().dt.days
    
    # 計算勝率
    total_trades = stock[buy_signal_col].sum() + stock[sell_signal_col].sum()
    win_trades = (stock['Strategy_Returns'] > 0).sum()
    win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0
    
    return stock, win_rate

# Streamlit應用程式主體
def main():
    st.title("股票技術分析工具")
    
    stockname = st.sidebar.text_input("輸入股票代號", value='AAPL')
    start_date = st.sidebar.date_input("選擇開始日期", value=pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("選擇結束日期", value=pd.to_datetime("2023-12-31"))
    interval = st.sidebar.selectbox("選擇數據頻率", options=['1d', '1wk', '1mo'], index=0)
    strategy_name = st.sidebar.selectbox("選擇交易策略", options=["Bollinger Bands", "KDJ", "RSI", "MACD", "唐奇安通道"], index=0)

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
            signals, win_rate = bollinger_band_strategy(stock, period=bollinger_period, std_dev=bollinger_std)
            st.write("策略回測績效:")
            st.write(f"勝率: {win_rate:.2f}%")
            st.write(signals.head())
            plot_stock_data(signals, strategy_name)
        elif strategy_name == "KDJ":
            signals, win_rate = kdj_strategy(stock, period=kdj_period)
            st.write("策略回測績效:")
            st.write(f"勝率: {win_rate:.2f}%")
            st.write(signals.head())
            plot_kdj(signals)
        elif strategy_name == "RSI":
            signals, win_rate = rsi_strategy(stock, period=rsi_period)
            st.write("策略回測績效:")
            st.write(f"勝率: {win_rate:.2f}%")
            st.write(signals.head())
            plot_rsi(signals)
        elif strategy_name == "MACD":
            signals, win_rate = macd_strategy(stock, short_window=short_window, long_window=long_window, signal_window=signal_window)
            st.write("策略回測績效:")
            st.write(f"勝率: {win_rate:.2f}%")
            st.write(signals.head())
            plot_macd(signals)
        elif strategy_name == "唐奇安通道":
            signals, win_rate = donchian_channel_strategy(stock, period=donchian_period)
            st.write("策略回測績效:")
            st.write(f"勝率: {win_rate:.2f}%")
            st.write(signals.head())
            plot_stock_data(signals, strategy_name)

if __name__ == '__main__':
    main()
