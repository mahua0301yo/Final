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

# 定義函數來計算KDJ指標
def calculate_kdj(stock, period=14, m=3):
    # 計算N日內最低價和最高價
    low_min = stock['Low'].rolling(window=period).min()
    high_max = stock['High'].rolling(window=period).max()

    # 計算RSV
    stock['RSV'] = (stock['Close'] - low_min) / (high_max - low_min) * 100

    # 計算K、D、J值
    stock['K'] = stock['RSV'].ewm(span=m).mean()
    stock['D'] = stock['K'].ewm(span=m).mean()
    stock['J'] = 3 * stock['K'] - 2 * stock['D']
    return stock

# 定義函數來繪製股票數據和技術指標
def plot_stock_data(stock, strategy_name):
    fig = go.Figure()
    
    # 添加 OHLC 圖
    fig.add_trace(go.Candlestick(x=stock['Date'],
                                 open=stock['Open'],
                                 high=stock['High'],
                                 low=stock['Low'],
                                 close=stock['Close'],
                                 name='OHLC'))
    
    # 添加成交量柱狀圖
    fig.add_trace(go.Bar(x=stock['Date'],
                         y=stock['Volume'],
                         name='Volume',
                         marker_color='lightgray',
                         yaxis='y2'))
    
    if strategy_name == "KDJ":
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['K'], mode='lines', name='K'))
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['D'], mode='lines', name='D'))
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['J'], mode='lines', name='J'))
    
    # 設定圖表的 layout
    fig.update_layout(
        title=f'{strategy_name} Analysis',
        xaxis_title='Date',
        yaxis_title='Price',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            showgrid=False,
            tickformat=','
        ),
        legend=dict(x=0, y=1.2),
        height=800,
    )
    
    st.plotly_chart(fig)

# 主函數
def main():
    st.sidebar.header("參數設定")
    stockname = st.sidebar.text_input("輸入股票代號", value="AAPL")
    start_date = st.sidebar.date_input("選擇開始日期", value=pd.to_datetime("2023-01-01"))
    end_date = st.sidebar.date_input("選擇結束日期", value=pd.to_datetime("2023-12-31"))
    interval = st.sidebar.selectbox("選擇數據頻率", options=['1d', '1wk', '1mo'], index=0)
    strategy_name = st.sidebar.selectbox("選擇交易策略", options=["KDJ"], index=0)

    if strategy_name == "KDJ":
        kdj_period = st.sidebar.slider("KDJ週期", min_value=5, max_value=50, value=14, step=1)

    # 讀取股票數據
    stock = load_stock_data(stockname, start_date, end_date, interval)
    if stock is not None:
        st.subheader(f"股票代號: {stockname}")
        st.write(stock.head())

        # 計算技術指標並繪圖
        if strategy_name == "KDJ":
            stock = calculate_kdj(stock, period=kdj_period)

        plot_stock_data(stock, strategy_name)

if __name__ == "__main__":
    main()
