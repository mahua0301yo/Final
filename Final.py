# 載入必要模組
import yfinance as yf
import pandas as pd
import streamlit as st
import streamlit.components.v1 as stc
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import datetime

# 定義函數來顯示標題
def display_header():
    html_temp = """
        <div style="background-color:#3872fb;padding:10px;border-radius:10px">
        <h1 style="color:white;text-align:center;">金融資料視覺化呈現 (金融看板) </h1>
        <h2 style="color:white;text-align:center;">Financial Dashboard </h2>
        </div>
    """
    stc.html(html_temp)

# 定義函數來讀取股票數據
def load_stock_data(stockname, start_date, end_date):
    try:
        stock = yf.download(stockname, start=start_date, end=end_date)
        if stock.empty:
            st.error("未能讀取到數據，請檢查股票代號是否正確")
            return None
        else:
            st.success("數據讀取成功")
            stock.rename(columns={'Volume': 'amount'}, inplace=True)
            stock.drop(columns=['Adj Close'], inplace=True)
            stock['Volume'] = (stock['amount'] / stock['Open']).astype(int)
            cols = stock.columns.tolist()
            vol_idx = cols.index('Volume')
            amt_idx = cols.index('amount')
            cols[vol_idx], cols[amt_idx] = cols[amt_idx], cols[vol_idx]
            stock = stock[cols]
            stock.reset_index(inplace=True)
            return stock
    except Exception as e:
        st.error(f"讀取數據時出錯: {e}")
        return None

# 定義函數來計算技術指標
def calculate_indicators(stock):
    stock['SMA_20'] = stock['Close'].rolling(window=20).mean()
    stock['EMA_20'] = stock['Close'].ewm(span=20, adjust=False).mean()
    return stock

# 定義函數來繪製圖表
def plot_stock_data(stock):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 繪製 K 線圖
    fig.add_trace(go.Candlestick(x=stock['Date'],
                                 open=stock['Open'], high=stock['High'],
                                 low=stock['Low'], close=stock['Close'], name='K線'),
                  secondary_y=True)
    
    # 繪製成交量
    fig.add_trace(go.Bar(x=stock['Date'], y=stock['Volume'], name='成交量', marker=dict(color='black')),
                  secondary_y=False)
    
    # 繪製 SMA 和 EMA
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['SMA_20'], mode='lines', name='SMA 20', line=dict(color='blue', width=2)),
                  secondary_y=True)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['EMA_20'], mode='lines', name='EMA 20', line=dict(color='orange', width=2)),
                  secondary_y=True)
    
    fig.layout.yaxis2.showgrid = True
    st.plotly_chart(fig, use_container_width=True)

# 主函數
def main():
    display_header()
    
    # 選擇資料區間
    st.subheader("選擇資料區間")
    start_date = st.date_input('選擇開始日期', datetime.date(2000, 1, 1))
    end_date = st.date_input('選擇結束日期', datetime.date(2100, 12, 31))
    stockname = st.text_input('請輸入股票代號 (例: 2330.TW)', '2330.TW')
    
    # 驗證日期輸入
    if start_date > end_date:
        st.error("開始日期不能晚於結束日期")
    else:
        stock = load_stock_data(stockname, start_date, end_date)
        if stock is not None:
            stock = calculate_indicators(stock)
            plot_stock_data(stock)

if __name__ == "__main__":
    main()
