import yfinance as yf
import pandas as pd
import streamlit as st
import streamlit.components.v1 as stc
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import datetime

def display_header():
    html_temp = """
        <div style="background-color:#3872fb;padding:10px;border-radius:10px">
        <h1 style="color:white;text-align:center;">金融資料視覺化呈現 (金融看板) </h1>
        <h2 style="color:white;text-align:center;">Financial Dashboard </h2>
        </div>
    """
    stc.html(html_temp)

def load_stock_data(stockname, start_date, end_date, interval):
    try:
        # Convert interval for 6 months and 1 year
        if interval == '6個月':
            interval = '1d'
            period = '6mo'
        elif interval == '1年':
            interval = '1d'
            period = '1y'
        else:
            period = None

        if period:
            stock = yf.download(stockname, period=period, interval=interval)
        else:
            stock = yf.download(stockname, start=start_date, end=end_date, interval=interval)

        if stock.empty:
            st.error("未能讀取到數據，請檢查股票代號是否正確")
            return None
        else:
            st.success("數據讀取成功")
            stock.rename(columns={'Volume': 'amount'}, inplace=True)
            stock.drop(columns=['Adj Close'], inplace=True)
            stock['Volume'] = (stock['amount'] / (stock['Open'] + stock['Close']) / 2).astype(int)
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

def calculate_indicators(stock, sma_period, ema_period):
    stock[f'SMA_{sma_period}'] = stock['Close'].rolling(window=sma_period).mean()
    stock[f'EMA_{ema_period}'] = stock['Close'].ewm(span=ema_period, adjust=False).mean()
    return stock

def plot_stock_data(stock, sma_period, ema_period):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 繪製 K 線圖
    fig.add_trace(go.Candlestick(x=stock['Date'],
                                 open=stock['Open'], high=stock['High'],
                                 low=stock['Low'], close=stock['Close'],
                                 name='K線'))

    # 繪製 SMA 和 EMA
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock[f'SMA_{sma_period}'], mode='lines', name=f'SMA {sma_period}'),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock[f'EMA_{ema_period}'], mode='lines', name=f'EMA {ema_period}'),
                  secondary_y=False)

    # 繪製成交量
    fig.add_trace(go.Bar(x=stock['Date'], y=stock['Volume'], name='成交量'), secondary_y=True)

    fig.update_layout(title='股票價格與技術指標', xaxis_title='日期', yaxis_title='價格', yaxis2_title='成交量')
    st.plotly_chart(fig)

def main():
    display_header()
    
    stockname = st.text_input("請輸入股票代號 (例如: 2330.TW)", "2330.TW")
    kline_interval = st.selectbox("選擇K線時間長", ["1個月", "3個月", "6個月", "1年"])
    sma_period = st.number_input("請輸入SMA週期", min_value=1, max_value=100, value=20)
    ema_period = st.number_input("請輸入EMA週期", min_value=1, max_value=100, value=20)

    # 根據選擇的K線時間長設置日期範圍
    today = datetime.date.today()
    if kline_interval == "1個月":
        start_date = today - datetime.timedelta(days=30)
    elif kline_interval == "3個月":
        start_date = today - datetime.timedelta(days=90)
    elif kline_interval == "6個月":
        start_date = today - datetime.timedelta(days=180)
    elif kline_interval == "1年":
        start_date = today - datetime.timedelta(days=365)
    end_date = today

    if st.button("加載數據"):
        stock = load_stock_data(stockname, start_date, end_date, kline_interval)
        if stock is not None:
            stock = calculate_indicators(stock, sma_period, ema_period)
            plot_stock_data(stock, sma_period, ema_period)

if __name__ == "__main__":
    main()
