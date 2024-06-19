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
    if macd_short_period >= 1:
        stock['EMA12'] = stock['Close'].ewm(span=macd_short_period).mean()
    else:
        raise ValueError("MACD短期EMA週期（macd_short_period）必須大於等於1")

    if macd_long_period >= 1:
        stock['EMA26'] = stock['Close'].ewm(span=macd_long_period).mean()
    else:
        raise ValueError("MACD長期EMA週期（macd_long_period）必須大於等於1")

    stock['MACD'] = stock['EMA12'] - stock['EMA26']

    if macd_signal_period >= 1:
        stock['MACD_Signal'] = stock['MACD'].ewm(span=macd_signal_period).mean()
    else:
        raise ValueError("MACD信號線週期（macd_signal_period）必須大於等於1")

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
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Middle_Band'], mode='lines', name='Middle Band'))
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Upper_Band'], mode='lines', name='Upper Band'))
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Lower_Band'], mode='lines', name='Lower Band'))
    elif strategy_name == "MACD":
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['MACD'], mode='lines', name='MACD'))
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['MACD_Signal'], mode='lines', name='MACD Signal'))
        fig.add_trace(go.Bar(x=stock['Date'], y=stock['MACD_Hist'], name='MACD Histogram'))

    fig.update_layout(title=f'{strategy_name} 策略', xaxis_title='日期', yaxis_title='價格', showlegend=True)
    st.plotly_chart(fig)

# 主函數
def main():
    display_header()

    # 用戶輸入部分
    st.sidebar.subheader("設置")
    stockname = st.sidebar.text_input("輸入股票代號", value='AAPL')
    start_date = st.sidebar.date_input("開始日期", datetime.date(2010, 1, 1))
    end_date = st.sidebar.date_input("結束日期", datetime.date(2023, 1, 1))
    interval = st.sidebar.selectbox("選擇數據頻率", options=['1d', '1wk', '1mo'], index=0)
    strategy_name = st.sidebar.selectbox("選擇交易策略", options=["KDJ", "Bollinger Bands", "MACD"], index=0)

    if strategy_name == "Bollinger Bands":
        bollinger_period = st.sidebar.slider("布林通道週期", min_value=5, max_value=50, value=20, step=1)
        bollinger_std = st.sidebar.slider("布林通道標準差倍數", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
        macd_short_period, macd_long_period, macd_signal_period = 0, 0, 0
    elif strategy_name == "MACD":
        macd_short_period = st.sidebar.number_input("輸入MACD短期EMA週期", min_value=1, max_value=50, value=12, step=1)
        macd_long_period = st.sidebar.number_input("輸入MACD長期EMA週期", min_value=1, max_value=50, value=26, step=1)
        macd_signal_period = st.sidebar.number_input("輸入MACD信號線週期", min_value=1, max_value=50, value=9, step=1)
        bollinger_period, bollinger_std = 0, 0
    else:
        bollinger_period, bollinger_std, macd_short_period, macd_long_period, macd_signal_period = 0, 0, 0, 0, 0

    # 讀取股票數據
    stock = load_stock_data(stockname, start_date, end_date, interval)
    if stock is not None:
        st.subheader(f"股票代號: {stockname}")
        st.write(stock.head())

        # 計算技術指標並繪圖
        if strategy_name == "KDJ":
            stock = calculate_kdj(stock)
            stock = kdj_strategy(stock)
        elif strategy_name == "Bollinger Bands":
            stock = calculate_indicators(stock, bollinger_period, bollinger_std, 0, 0, 0)
            stock = bollinger_strategy(stock)
        elif strategy_name == "MACD":
            stock = calculate_indicators(stock, 0, 0, macd_short_period, macd_long_period, macd_signal_period)
            stock = macd_strategy(stock)

        plot_stock_data(stock, strategy_name)

        # 顯示績效指標
        performance_metrics = calculate_performance(stock)
        st.subheader("績效指標")
        st.write(performance_metrics)

if __name__ == "__main__":
    main()
