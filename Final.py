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
    delta = stock['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    stock['RSI'] = 100 - (100 / (1 + rs))
    
    # 加入超買超賣區域設定
    stock['Overbought'] = 80
    stock['Oversold'] = 20
    
    return stock

# 定義函數來繪製股票數據和技術指標
def plot_stock_data(stock, strategy_name):
    fig = go.Figure()

    # 繪製 K 線圖
    fig.add_trace(go.Candlestick(
        x=stock['Date'],
        open=stock['Open'],
        high=stock['High'],
        low=stock['Low'],
        close=stock['Close'],
        name='OHLC'
    ))

    if strategy_name == 'Bollinger Bands':
        fig.add_trace(go.Scatter(
            x=stock['Date'],
            y=stock['Middle_Band'],
            mode='lines',
            name='Middle Band'
        ))
        fig.add_trace(go.Scatter(
            x=stock['Date'],
            y=stock['Upper_Band'],
            mode='lines',
            name='Upper Band'
        ))
        fig.add_trace(go.Scatter(
            x=stock['Date'],
            y=stock['Lower_Band'],
            mode='lines',
            name='Lower Band'
        ))
    elif strategy_name == 'MACD':
        # 繪製 12 週期和 26 週期的 EMA
        fig.add_trace(go.Scatter(
            x=stock['Date'],
            y=stock['Short_EMA'],
            mode='lines',
            name='12-period EMA',
            line=dict(color='magenta')
        ))
        fig.add_trace(go.Scatter(
            x=stock['Date'],
            y=stock['Long_EMA'],
            mode='lines',
            name='26-period EMA',
            line=dict(color='purple')
        ))
        # 添加 MACD 和 Signal Line 圖
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(
            x=stock['Date'],
            y=stock['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='blue')
        ))
        macd_fig.add_trace(go.Scatter(
            x=stock['Date'],
            y=stock['Signal_Line'],
            mode='lines',
            name='MACD Signal Line',
            line=dict(color='red')
        ))
        macd_fig.add_trace(go.Bar(
            x=stock['Date'],
            y=stock['MACD_Histogram'],
            name='MACD Histogram',
            marker_color='grey'
        ))
        macd_fig.update_layout(
            title='MACD',
            xaxis_title='Date',
            yaxis_title='MACD Value'
        )
        st.plotly_chart(macd_fig)
    elif strategy_name == 'KDJ':
        fig.add_trace(go.Scatter(
            x=stock['Date'],
            y=stock['K'],
            mode='lines',
            name='K',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=stock['Date'],
            y=stock['D'],
            mode='lines',
            name='D',
            line=dict(color='orange')
        ))
        fig.add_trace(go.Scatter(
            x=stock['Date'],
            y=stock['J'],
            mode='lines',
            name='J',
            line=dict(color='purple')
        ))
    elif strategy_name == 'RSI':
        fig.add_trace(go.Scatter(
            x=stock['Date'],
            y=stock['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='green')
        ))
        fig.add_trace(go.Scatter(
            x=stock['Date'],
            y=stock['Overbought'],
            mode='lines',
            name='Overbought',
            line=dict(color='red', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=stock['Date'],
            y=stock['Oversold'],
            mode='lines',
            name='Oversold',
            line=dict(color='blue', dash='dash')
        ))

    fig.update_layout(title=f'{strategy_name} Analysis',
                      xaxis_title='Date',
                      yaxis_title='Price')
    st.plotly_chart(fig)

# Streamlit 應用程式
def main():
    st.title('股票技術指標分析')

    # 用戶輸入
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

        # 計算技術指標並繪圖
        if strategy_name == "Bollinger Bands":
            stock = calculate_bollinger_bands(stock, period=bollinger_period, std_dev=bollinger_std)
        elif strategy_name == "MACD":
            stock = calculate_macd(stock, short_period=macd_short_period, long_period=macd_long_period, signal_period=macd_signal_period)
        elif strategy_name == "KDJ":
            stock = calculate_kdj(stock, period=kdj_period)
        elif strategy_name == "RSI":
            stock = calculate_rsi(stock, period=rsi_period)

        plot_stock_data(stock, strategy_name)

if __name__ == "__main__":
    main()
