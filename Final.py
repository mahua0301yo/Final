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
def calculate_kdj(stock, period=14, m=3):
    low_min = stock['Low'].rolling(window=period).min()
    high_max = stock['High'].rolling(window=period).max()
    stock['RSV'] = (stock['Close'] - low_min) / (high_max - low_min) * 100
    stock['K'] = stock['RSV'].ewm(span=m).mean()
    stock['D'] = stock['K'].ewm(span=m).mean()
    stock['J'] = 3 * stock['K'] - 2 * stock['D']
    return stock

# 定義函數來計算RSI指標
def calculate_rsi(stock, period=14):
    delta = stock['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    stock['RSI'] = 100 - (100 / (1 + rs))
    return stock

# 繪製股票數據及技術指標
def plot_stock_data(stock, strategy_name, price_range):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=stock['Date'], open=stock['Open'], high=stock['High'], low=stock['Low'], close=stock['Close'],
        name='Candlestick'))

    if strategy_name == "Bollinger Bands":
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Upper_Band'], line=dict(color='blue', width=1), name='Upper Band'))
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Middle_Band'], line=dict(color='orange', width=1), name='Middle Band'))
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Lower_Band'], line=dict(color='blue', width=1), name='Lower Band'))
    elif strategy_name == "MACD":
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['MACD'], line=dict(color='blue', width=1), name='MACD'))
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Signal_Line'], line=dict(color='red', width=1), name='Signal Line'))
        fig.add_trace(go.Bar(x=stock['Date'], y=stock['MACD_Histogram'], name='MACD Histogram'))
    elif strategy_name == "KDJ":
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['K'], line=dict(color='blue', width=1), name='K'))
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['D'], line=dict(color='orange', width=1), name='D'))
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['J'], line=dict(color='green', width=1), name='J'))
    elif strategy_name == "RSI":
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['RSI'], line=dict(color='blue', width=1), name='RSI'))

    fig.update_layout(title=strategy_name, xaxis_title='Date', yaxis_title='Price', yaxis=dict(range=price_range))
    st.plotly_chart(fig)

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

    # 讀取股票數據
    stock = load_stock_data(stockname, start_date, end_date, interval)
    if stock is not None:
        st.subheader(f"股票代號: {stockname}")
        st.write(stock.head())

        # 計算價格範圍
        min_price = stock['Low'].min()
        max_price = stock['High'].max()
        price_range = st.sidebar.slider(
            "價格範圍",
            min_value=float(min_price * 0.9),
            max_value=float(max_price * 1.1),
            value=(float(min_price * 0.9), float(max_price * 1.1)),
            step=0.1
        )

        # 計算技術指標並繪圖
        if strategy_name == "Bollinger Bands":
            stock = calculate_bollinger_bands(stock, period=bollinger_period, std_dev=bollinger_std)
        elif strategy_name == "MACD":
            stock = calculate_macd(stock, short_period=macd_short_period, long_period=macd_long_period, signal_period=macd_signal_period)
        elif strategy_name == "KDJ":
            stock = calculate_kdj(stock, period=kdj_period)
        elif strategy_name == "RSI":
            stock = calculate_rsi(stock, period=rsi_period)

        plot_stock_data(stock, strategy_name, price_range)

if __name__ == "__main__":
    main()
