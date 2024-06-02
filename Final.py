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
def load_stock_data(stockname, start_date, end_date, interval):
    try:
        stock = yf.download(stockname, start=start_date, end=end_date, interval="1d")
        if stock.empty:
            st.error("未能讀取到數據，請檢查股票代號是否正確")
            return None
        else:
            st.success("數據讀取成功")
            stock.rename(columns={'Volume': 'Amount'}, inplace=True)
            stock.drop(columns=['Adj Close'], inplace=True)
            stock.reset_index(inplace=True)
            stock['Date'] = pd.to_datetime(stock['Date'])
            stock.set_index('Date', inplace=True)
            return stock
    except Exception as e:
        st.error(f"讀取數據時出錯: {e}")
        return None

# 定義函數來重新取樣數據
def resample_data(stock, interval):
    if interval == "1d":
        return stock
    else:
        resample_rule = {
            "3mo": 'Q',
            "6mo": '2Q',
            "1y": 'A'
        }
        stock = stock.resample(resample_rule[interval]).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        return stock

# 定義函數來計算技術指標
def calculate_indicators(stock, sma_period, ema_period):
    stock[f'SMA_{sma_period}'] = stock['Close'].rolling(window=sma_period).mean()
    stock[f'EMA_{ema_period}'] = stock['Close'].ewm(span=ema_period, adjust=False).mean()
    return stock

# 定義函數來繪製圖表
def plot_stock_data(stock, sma_period, ema_period):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 繪製 K 線圖
    fig.add_trace(go.Candlestick(x=stock.index,
                                 open=stock['Open'], high=stock['High'],
                                 low=stock['Low'], close=stock['Close'], name='K線'),
                  secondary_y=True)
    
    # 繪製成交量
    fig.add_trace(go.Bar(x=stock.index, y=stock['Volume'], name='成交量', marker=dict(color='black')),
                  secondary_y=False)
    
    # 繪製 SMA 和 EMA
    fig.add_trace(go.Scatter(x=stock.index, y=stock[f'SMA_{sma_period}'], mode='lines', name=f'SMA {sma_period}', line=dict(color='blue', width=2)),
                  secondary_y=True)
    fig.add_trace(go.Scatter(x=stock.index, y=stock[f'EMA_{ema_period}'], mode='lines', name=f'EMA {ema_period}', line=dict(color='orange', width=2)),
                  secondary_y=True)
    
    # 調整日期軸格式
    fig.update_xaxes(rangeslider_visible=True,
                     rangeselector=dict(
                         buttons=list([
                             dict(count=1, label="1m", step="month", stepmode="backward"),
                             dict(count=6, label="6m", step="month", stepmode="backward"),
                             dict(count=1, label="YTD", step="year", stepmode="todate"),
                             dict(count=1, label="1y", step="year", stepmode="backward"),
                             dict(step="all")
                         ])
                     ))

    # 調整布局
    fig.update_layout(
        title_text='股票價格與技術指標',
        yaxis_title='成交量',
        yaxis2_title='價格',
        xaxis_title='日期',
        xaxis_rangeslider_visible=False,
        legend=dict(x=0, y=1, traceorder='normal')
    )

    st.plotly_chart(fig, use_container_width=True)

# 主函數
def main():
    display_header()
    
    # 選擇資料區間
    st.subheader("選擇資料區間")
    start_date = st.date_input('選擇開始日期', datetime.date(2000, 1, 1))
    end_date = st.date_input('選擇結束日期', datetime.date(2100, 12, 31))
    stockname = st.text_input('請輸入股票代號 (例: 2330.TW)', '2330.TW')
    
    # 選擇K線時間長
    interval_options = {
        "1天": "1d",
        "1星期": "1wk",
        "1個月": "1mo",
        "3個月": "3mo",
        "6個月": "6mo",
        "1年": "1y"
    }
    interval_label = st.selectbox("選擇K線時間長", list(interval_options.keys()))
    interval = interval_options[interval_label]
    
    # 輸入 SMA 和 EMA 的週期
    sma_period = st.number_input('請輸入SMA週期', min_value=1, max_value=100, value=20, step=1)
    ema_period = st.number_input('請輸入EMA週期', min_value=1, max_value=100, value=20, step=1)
    
    # 驗證日期輸入
    if start_date > end_date:
        st.error("開始日期不能晚於結束日期")
    else:
        stock = load_stock_data(stockname, start_date, end_date, interval)
        if stock is not None:
            stock = resample_data(stock, interval)
            stock = calculate_indicators(stock, sma_period, ema_period)
            plot_stock_data(stock, sma_period, ema_period)

if __name__ == "__main__":
    main()
