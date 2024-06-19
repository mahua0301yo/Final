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

# 定義函數來計算KDJ指標
def calculate_kdj(stock, period=14):
    low_min = stock['Low'].rolling(window=period).min()
    high_max = stock['High'].rolling(window=period).max()
    rsv = (stock['Close'] - low_min) / (high_max - low_min) * 100
    stock['K'] = rsv.ewm(com=2).mean()
    stock['D'] = stock['K'].ewm(com=2).mean()
    stock['J'] = 3 * stock['K'] - 2 * stock['D']
    return stock

# 定義函數來計算布林通道和MACD指標
def calculate_indicators(stock, bollinger_period, bollinger_std, macd_short_period, macd_long_period, macd_signal_period):
    # 計算布林通道
    stock['Middle_Band'] = stock['Close'].rolling(window=bollinger_period).mean()
    stock['Upper_Band'] = stock['Middle_Band'] + bollinger_std * stock['Close'].rolling(window=bollinger_period).std()
    stock['Lower_Band'] = stock['Middle_Band'] - bollinger_std * stock['Close'].rolling(window=bollinger_period).std()

    # 計算MACD
    stock['EMA12'] = stock['Close'].ewm(span=macd_short_period).mean()
    stock['EMA26'] = stock['Close'].ewm(span=macd_long_period).mean()
    stock['MACD'] = stock['EMA12'] - stock['EMA26']
    stock['MACD_Signal'] = stock['MACD'].ewm(span=macd_signal_period).mean()
    stock['MACD_Hist'] = stock['MACD'] - stock['MACD_Signal']
    
    return stock

# 定義交易策略函數
def kdj_strategy(stock):
    stock['Signal'] = 0
    stock['Signal'] = np.where((stock['K'] > stock['D']) & (stock['K'].shift(1) <= stock['D'].shift(1)), 1, stock['Signal'])
    stock['Signal'] = np.where((stock['K'] < stock['D']) & (stock['K'].shift(1) >= stock['D'].shift(1)), -1, stock['Signal'])
    return stock

def bollinger_strategy(stock):
    stock['Signal'] = 0
    stock['Signal'] = np.where(stock['Close'] < stock['Lower_Band'], 1, stock['Signal'])
    stock['Signal'] = np.where(stock['Close'] > stock['Upper_Band'], -1, stock['Signal'])
    return stock

def macd_strategy(stock):
    stock['Signal'] = 0
    stock['Signal'] = np.where((stock['MACD'] > stock['MACD_Signal']) & (stock['MACD'].shift(1) <= stock['MACD_Signal'].shift(1)), 1, stock['Signal'])
    stock['Signal'] = np.where((stock['MACD'] < stock['MACD_Signal']) & (stock['MACD'].shift(1) >= stock['MACD_Signal'].shift(1)), -1, stock['Signal'])
    return stock

# 計算績效指標
def calculate_performance(stock):
    initial_cash = 1000000  # 初始資金
    cash = initial_cash
    position = 0
    trades = []
    
    for index, row in stock.iterrows():
        if row['Signal'] == 1 and cash > 0:
            position = cash / row['Close']
            cash = 0
        elif row['Signal'] == -1 and position > 0:
            cash = position * row['Close']
            position = 0
            trades.append(cash - initial_cash)
    
    total_profit = cash + (position * stock.iloc[-1]['Close']) - initial_cash
    win_rate = sum(1 for trade in trades if trade > 0) / len(trades) if trades else 0
    max_consecutive_loss = calculate_max_consecutive_losses(trades)
    max_drawdown = calculate_max_drawdown(stock, initial_cash)
    return_rate = (total_profit / initial_cash) * 100
    
    performance_metrics = {
        'Total Profit': total_profit,
        'Win Rate': win_rate,
        'Max Consecutive Losses': max_consecutive_loss,
        'Max Drawdown': max_drawdown,
        'Return Rate': return_rate
    }
    
    return performance_metrics

# 計算最大連續虧損
def calculate_max_consecutive_losses(trades):
    max_consecutive_loss = 0
    current_loss_streak = 0
    
    for trade in trades:
        if trade < 0:
            current_loss_streak += 1
        else:
            if current_loss_streak > max_consecutive_loss:
                max_consecutive_loss = current_loss_streak
            current_loss_streak = 0
    
    if current_loss_streak > max_consecutive_loss:
        max_consecutive_loss = current_loss_streak
    
    return max_consecutive_loss

# 計算最大回撤
def calculate_max_drawdown(stock, initial_cash):
    peak = initial_cash
    max_drawdown = 0
    cash = initial_cash
    position = 0
    
    for index, row in stock.iterrows():
        if row['Signal'] == 1 and cash > 0:
            position = cash / row['Close']
            cash = 0
        elif row['Signal'] == -1 and position > 0:
            cash = position * row['Close']
            position = 0
        
        current_value = cash + (position * row['Close'])
        if current_value > peak:
            peak = current_value
        drawdown = (peak - current_value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    return max_drawdown * 100  # Convert to percentage

# 定義繪製股票數據的函數
def plot_stock_data(stock, strategy_name):
    fig = go.Figure()

    # 繪製收盤價
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Close'], mode='lines', name='Close'))

    if strategy_name == "KDJ":
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['K'], mode='lines', name='K'))
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['D'], mode='lines', name='D'))
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['J'], mode='lines', name='J'))
    elif strategy_name == "Bollinger Bands":
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Upper_Band'], mode='lines', name='Upper Band'))
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Lower_Band'], mode='lines', name='Lower Band'))
    elif strategy_name == "MACD":
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['MACD'], mode='lines', name='MACD'))
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['MACD_Signal'], mode='lines', name='MACD Signal'))
        fig.add_trace(go.Bar(x=stock['Date'], y=stock['MACD_Hist'], name='MACD Hist'))
    
    st.plotly_chart(fig)

    # 計算和顯示績效指標
    performance_metrics = calculate_performance(stock)
    st.write("績效指標:")
    st.write(f"總損益: {performance_metrics['Total Profit']}")
    st.write(f"勝率: {performance_metrics['Win Rate']}")
    st.write(f"最大連續虧損: {performance_metrics['Max Consecutive Losses']}")
    st.write(f"最大資金回落 (MDD): {performance_metrics['Max Drawdown']}%")
    st.write(f"報酬率: {performance_metrics['Return Rate']}%")

# 主函數
def main():
    display_header()

    # 選擇資料區間
    st.subheader("選擇資料區間")
    start_date = st.date_input('選擇開始日期', datetime.date(2015, 1, 1))
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

    # 選擇技術指標
    strategy_options = ["KDJ", "Bollinger Bands", "MACD"]
    strategy_name = st.selectbox("選擇技術指標", strategy_options)

    # 設定指標參數
    if strategy_name == "KDJ":
        kdj_period = st.number_input('請輸入KDJ週期', min_value=1, max_value=100, value=14, step=1)
    elif strategy_name == "Bollinger Bands":
        bollinger_period = st.number_input('請輸入布林通道週期', min_value=1, max_value=100, value=20, step=1)
        bollinger_std = st.number_input('請輸入布林通道標準差倍數', min_value=0.1, max_value=10.0, value=2.0, step=0.1)
    elif strategy_name == "MACD":
        macd_short_period = st.number_input('請輸入MACD短期EMA週期', min_value=1, max_value=50, value=12, step=1)
        macd_long_period = st.number_input('請輸入MACD長期EMA週期', min_value=1, max_value=50, value=26, step=1)
        macd_signal_period = st.number_input('請輸入MACD信號線週期', min_value=1, max_value=50, value=9, step=1)

    # 驗證日期輸入
    if start_date > end_date:
        st.error("開始日期不能晚於結束日期")
    else:
        stock = load_stock_data(stockname, start_date, end_date, interval)
        if stock is not None:
            if strategy_name == "KDJ":
                stock = calculate_kdj(stock, kdj_period)
                stock = kdj_strategy(stock)
            elif strategy_name == "Bollinger Bands":
                stock = calculate_indicators(stock, bollinger_period, bollinger_std, 0, 0, 0)
                stock = bollinger_strategy(stock)
            elif strategy_name == "MACD":
                stock = calculate_indicators(stock, 0, 0, macd_short_period, macd_long_period, macd_signal_period)
                stock = macd_strategy(stock)
            
            plot_stock_data(stock, strategy_name)

if __name__ == "__main__":
    main()
