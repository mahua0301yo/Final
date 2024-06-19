# 載入必要模組
import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objs as go

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

# 定義函數來計算MACD指標
def calculate_macd(stock, short_period=12, long_period=26, signal_period=9):
    stock['Short_EMA'] = stock['Close'].ewm(span=short_period, adjust=False).mean()
    stock['Long_EMA'] = stock['Close'].ewm(span=long_period, adjust=False).mean()
    stock['MACD'] = stock['Short_EMA'] - stock['Long_EMA']
    stock['Signal_Line'] = stock['MACD'].ewm(span=signal_period, adjust=False).mean()
    stock['MACD_Histogram'] = stock['MACD'] - stock['Signal_Line']
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
    return stock

# 繪製股票數據的主函數
def plot_stock_data(stock, strategy_name):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=stock['Date'], open=stock['Open'], high=stock['High'], low=stock['Low'], close=stock['Close'], name='OHLC'))

    if strategy_name == "Bollinger Bands":
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Upper_Band'], line=dict(color='red', width=1), name='Upper Band'))
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Middle_Band'], line=dict(color='blue', width=1), name='Middle Band'))
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Lower_Band'], line=dict(color='red', width=1), name='Lower Band'))

    fig.update_layout(title=f"{strategy_name} 指標圖", xaxis_title='日期', yaxis_title='價格', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

# 繪製MACD圖表，保留K線圖
def plot_macd(stock):
    fig = go.Figure()

    # K線圖
    fig.add_trace(go.Candlestick(x=stock['Date'], open=stock['Open'], high=stock['High'], low=stock['Low'], close=stock['Close'], name='OHLC'))

    # MACD及信號線
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['MACD'], line=dict(color='blue', width=1), name='MACD'))
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Signal_Line'], line=dict(color='red', width=1), name='Signal Line'))

    fig.update_layout(title="MACD 指標圖", xaxis_title='日期', yaxis_title='價格', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

# 繪製KDJ圖表
def plot_kdj(stock):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=stock['Date'], open=stock['Open'], high=stock['High'], low=stock['Low'], close=stock['Close'], name='OHLC'))
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['K'], line=dict(color='blue', width=1), name='K'))
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['D'], line=dict(color='red', width=1), name='D'))
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['J'], line=dict(color='green', width=1), name='J'))

    fig.update_layout(title="KDJ 指標圖", xaxis_title='日期', yaxis_title='KDJ 值', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

# 繪製RSI圖表
def plot_rsi(stock):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=stock['Date'], open=stock['Open'], high=stock['High'], low=stock['Low'], close=stock['Close'], name='OHLC'))
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['RSI'], line=dict(color='blue', width=1), name='RSI'))
    fig.add_shape(type="line", x0=stock['Date'].min(), y0=80, x1=stock['Date'].max(), y1=80, line=dict(color="red", width=2, dash="dash"))
    fig.add_shape(type="line", x0=stock['Date'].min(), y0=20, x1=stock['Date'].max(), y1=20, line=dict(color="green", width=2, dash="dash"))

    fig.update_layout(title="RSI 指標圖", xaxis_title='日期', yaxis_title='RSI 值', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

# 主函數
def main():
    st.title("股票技術指標分析")

    stockname = st.sidebar.text_input("輸入股票代號", value='AAPL')
    start_date = st.sidebar.date_input("選擇開始日期", value=pd.to_datetime("2023-01-01"))
    end_date = st.sidebar.date_input("選擇結束日期", value=pd.to_datetime("2023-12-31"))
    interval = st.sidebar.selectbox("選擇數據頻率", options=['1d', '1wk', '1mo'], index=0)
    strategy_name = st.sidebar.selectbox("選擇交易策略", options=["Bollinger Bands", "MACD", "KDJ", "RSI"], index=0)

    if strategy_name == "Bollinger Bands":
        bollinger_period = st.sidebar.slider("布林通道週期", min_value=5, max_value=50, value=20, step=1)
        bollinger_std = st.sidebar.slider("布林通道標準差倍數", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    elif strategy_name == "MACD":
        macd_short_period = st.sidebar.number_input("輸入MACD短期EMA週期", min_value=1, max_value=50, value=12, step=1)
        macd_long_period = st.sidebar.number_input("輸入MACD長期EMA週期", min_value=1, max_value=50, value=26, step=1)
        macd_signal_period = st.sidebar.number_input("輸入MACD信號線週期", min_value=1, max_value=50, value=9, step=1)
    elif strategy_name == "KDJ":
        kdj_period = st.sidebar.slider("KDJ週期", min_value=5, max_value=50, value=14, step=1)
    elif strategy_name == "RSI":
        rsi_period = st.sidebar.slider("RSI週期", min_value=5, max_value=50, value=14, step=1)

    stock = load_stock_data(stockname, start_date, end_date, interval)
    if stock is not None:
        st.subheader(f"股票代號: {stockname}")
        st.write(stock.head())

        if strategy_name == "Bollinger Bands":
            stock = calculate_bollinger_bands(stock, period=bollinger_period, std_dev=bollinger_std)
            plot_stock_data(stock, strategy_name)
        elif strategy_name == "MACD":
            stock = calculate_macd(stock, short_period=macd_short_period, long_period=macd_long_period, signal_period=macd_signal_period)
            plot_macd(stock)
        elif strategy_name == "KDJ":
            stock = calculate_kdj(stock, period=kdj_period)
            plot_kdj(stock)
        elif strategy_name == "RSI":
            stock = calculate_rsi(stock, period=rsi_period)
            plot_rsi(stock)

if __name__ == "__main__":
    main()
