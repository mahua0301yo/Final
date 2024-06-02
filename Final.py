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
        <h1 style="color:white;text-align:center;">02 </h1>
        <h2 style="color:white;text-align:center;">002</h2>
        </div>
    """
    stc.html(html_temp)

# 定義函數來讀取股票數據
def load_stock_data(stockname, start_date, end_date, interval):
    try:
        stock = yf.download(stockname, start=start_date, end=end_date, interval=interval)
        if stock.empty:
            st.error("未能讀取到數據，請檢查股票代號是否正確")
            return None
        else:
            st.success("數據讀取成功")
            stock.rename(columns={'Volume': 'amount'}, inplace=True)
            stock.drop(columns=['Adj Close'], inplace=True)
            stock['Volume'] = (stock['amount'] / (stock['Open']+stock['Close'])/2).astype(int)
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

# 定義函數來計算技術指標，包括布林通道
def calculate_indicators(stock, sma_period, ema_period, bollinger_period, bollinger_std):
    stock[f'SMA_{sma_period}'] = stock['Close'].rolling(window=sma_period).mean()
    stock[f'EMA_{ema_period}'] = stock['Close'].ewm(span=ema_period, adjust=False).mean()

    # 計算布林通道
    stock['Middle_Band'] = stock['Close'].rolling(window=bollinger_period).mean()
    stock['Upper_Band'] = stock['Middle_Band'] + (stock['Close'].rolling(window=bollinger_period).std() * bollinger_std)
    stock['Lower_Band'] = stock['Middle_Band'] - (stock['Close'].rolling(window=bollinger_period).std() * bollinger_std)
    
    return stock

# 定義函數來繪製圖表
def plot_stock_data(stock, sma_period, ema_period):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 繪製 K 線圖
    fig.add_trace(go.Candlestick(x=stock['Date'],
                                 open=stock['Open'],
                                 high=stock['High'],
                                 low=stock['Low'],
                                 close=stock['Close'],
                                 name='價格'),
                  secondary_y=True)

    # 繪製 SMA 和 EMA
    fig.add_trace(go.Scatter(x=stock['Date'],
                             y=stock[f'SMA_{sma_period}'],
                             mode='lines',
                             name=f'SMA_{sma_period}'), secondary_y=True)

    fig.add_trace(go.Scatter(x=stock['Date'],
                             y=stock[f'EMA_{ema_period}'],
                             mode='lines',
                             name=f'EMA_{ema_period}'), secondary_y=True)

    # 繪製布林通道
    fig.add_trace(go.Scatter(x=stock['Date'],
                             y=stock['Upper_Band'],
                             mode='lines',
                             line=dict(color='rgba(255, 0, 0, 0.5)'),
                             name='上軌'), secondary_y=True)
    fig.add_trace(go.Scatter(x=stock['Date'],
                             y=stock['Middle_Band'],
                             mode='lines',
                             line=dict(color='rgba(0, 0, 255, 0.5)'),
                             name='中軌'), secondary_y=True)
    fig.add_trace(go.Scatter(x=stock['Date'],
                             y=stock['Lower_Band'],
                             mode='lines',
                             line=dict(color='rgba(0, 255, 0, 0.5)'),
                             name='下軌'), secondary_y=True)

    # 繪製成交量
    fig.add_trace(go.Bar(x=stock['Date'],
                         y=stock['amount'],
                         name='成交量'), secondary_y=False)

    # 調整 X 軸和 Y 軸標籤
    fig.update_xaxes(title_text='日期')
    fig.update_yaxes(title_text='成交量', secondary_y=False)
    fig.update_yaxes(title_text='價格', secondary_y=True)

    # 更新布局和樣式
    fig.update_layout(title='股票價格與技術指標',
                      xaxis_rangeslider_visible=True,
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
        "3個月": "3mo"
    }
    interval_label = st.selectbox("選擇K線時間長", list(interval_options.keys()))
    interval = interval_options[interval_label]

    # 輸入 SMA 和 EMA 的週期
    sma_period = st.number_input('請輸入SMA週期', min_value=1, max_value=100, value=20, step=1)
    ema_period = st.number_input('請輸入EMA週期', min_value=1, max_value=100, value=20, step=1)

    # 輸入布林通道的週期和標準差倍數
    bollinger_period = st.number_input('請輸入布林通道週期', min_value=1, max_value=100, value=20, step=1)
    bollinger_std = st.number_input('請輸入布林通道標準差倍數', min_value=0.1, max_value=10.0, value=2.0, step=0.1)

    # 驗證日期輸入
    if start_date > end_date:
        st.error("開始日期不能晚於結束日期")
    else:
        stock = load_stock_data(stockname, start_date, end_date, interval)
        if stock is not None:
            stock = calculate_indicators(stock, sma_period, ema_period, bollinger_period, bollinger_std)
            plot_stock_data(stock, sma_period, ema_period)

if __name__ == "__main__":
    main()
