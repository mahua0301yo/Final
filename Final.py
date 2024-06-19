# 載入必要模組
import yfinance as yf
import pandas as pd
import streamlit as st
import streamlit.components.v1 as stc
import plotly.graph_objs as go
import datetime
import numpy as np

# 定義函數來顯示標題
def display_header():
    html_temp = """
        <div style="background-color:#3872fb;padding:10px;border-radius:10px">
        <h1 style="color:white;text-align:center;">股票技術指標分析</h1>
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

# 定義績效計算函數
def calculate_performance(stock, buy_signals, sell_signals):
    # 總損益
    profits = sell_signals.values - buy_signals.values
    total_profit = profits.sum()
    # 勝率
    win_rate = (profits > 0).mean()
    # 最大連續虧損
    max_consecutive_loss = (profits < 0).astype(int).groupby(profits.cumsum()).cumsum().max()
    # 最大資金回落
    cumulative_profits = profits.cumsum()
    max_drawdown = (cumulative_profits.cummax() - cumulative_profits).max()
    # 報酬率
    return_rate = total_profit / buy_signals.sum()
    return total_profit, win_rate, max_consecutive_loss, max_drawdown, return_rate

# 定義函數來計算KDJ指標
def calculate_kdj(stock, period=14):
    low_min = stock['Low'].rolling(window=period).min()
    high_max = stock['High'].rolling(window=period).max()
    rsv = (stock['Close'] - low_min) / (high_max - low_min) * 100
    stock['K'] = rsv.ewm(com=2).mean()
    stock['D'] = stock['K'].ewm(com=2).mean()
    stock['J'] = 3 * stock['K'] - 2 * stock['D']
    return stock

# 定義函數來計算技術指標，包括布林通道和MACD
def calculate_indicators(stock, bollinger_period, bollinger_std, macd_short_period, macd_long_period, macd_signal_period):
    # 計算布林通道
    stock['Middle_Band'] = stock['Close'].rolling(window=bollinger_period).mean()
    stock['Upper_Band'] = stock['Middle_Band'] + (bollinger_std * stock['Close'].rolling(window=bollinger_period).std())
    stock['Lower_Band'] = stock['Middle_Band'] - (bollinger_std * stock['Close'].rolling(window=bollinger_period).std())

    # 計算MACD
    stock['MACD'] = stock['Close'].ewm(span=macd_short_period, adjust=False).mean() - stock['Close'].ewm(span=macd_long_period, adjust=False).mean()
    stock['Signal_Line'] = stock['MACD'].ewm(span=macd_signal_period, adjust=False).mean()
    return stock

# 計算總績效
def calculate_overall_performance(stock, indicators):
    overall_performance = {}
    for indicator in indicators:
        buy_signals = stock[stock[indicator] > stock[indicator].shift(1)]
        sell_signals = stock[stock[indicator] < stock[indicator].shift(1)]
        overall_performance[indicator] = calculate_performance(stock, buy_signals, sell_signals)
    return overall_performance

# Streamlit app 設置
def main():
    st.title("股票技術指標分析")
    display_header()
    
    # 用戶輸入
    stockname = st.text_input("輸入股票代號", "AAPL")
    start_date = st.date_input("選擇開始日期", datetime.date(2020, 1, 1))
    end_date = st.date_input("選擇結束日期", datetime.date.today())
    interval = st.selectbox("選擇時間間隔", ["1d", "1wk", "1mo"])
    bollinger_period = st.number_input("布林通道週期", min_value=1, max_value=50, value=20)
    bollinger_std = st.number_input("布林通道標準差", min_value=1, max_value=5, value=2)
    macd_short_period = st.number_input("MACD短期週期", min_value=1, max_value=50, value=12)
    macd_long_period = st.number_input("MACD長期週期", min_value=1, max_value=50, value=26)
    macd_signal_period = st.number_input("MACD訊號週期", min_value=1, max_value=50, value=9)
    
    if st.button("讀取數據"):
        stock = load_stock_data(stockname, start_date, end_date, interval)
        if stock is not None:
            stock = calculate_kdj(stock)
            stock = calculate_indicators(stock, bollinger_period, bollinger_std, macd_short_period, macd_long_period, macd_signal_period)
            
            indicators = ['K', 'D', 'J', 'Middle_Band', 'Upper_Band', 'Lower_Band', 'MACD', 'Signal_Line']
            overall_performance = calculate_overall_performance(stock, indicators)
            
            for indicator, performance in overall_performance.items():
                total_profit, win_rate, max_consecutive_loss, max_drawdown, return_rate = performance
                st.write(f"指標: {indicator}")
                st.write(f"總損益: {total_profit}")
                st.write(f"勝率: {win_rate}")
                st.write(f"最大連續虧損: {max_consecutive_loss}")
                st.write(f"最大資金回落: {max_drawdown}")
                st.write(f"報酬率: {return_rate}")
                
            # 繪圖
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=stock['Date'],
                                         open=stock['Open'],
                                         high=stock['High'],
                                         low=stock['Low'],
                                         close=stock['Close'], name='Candlesticks'))
            fig.update_layout(title=f'{stockname} 股價走勢',
                              yaxis_title='股價',
                              xaxis_title='日期')
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
