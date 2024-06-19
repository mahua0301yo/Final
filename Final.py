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

# 定義交易策略函數
def kdj_strategy(stock):
    stock['Signal'] = 0
    stock['Signal'] = np.where((stock['K'] > stock['D']) & (stock['K'].shift(1) <= stock['D'].shift(1)), 1, stock['Signal'])
    stock['Signal'] = np.where((stock['K'] < stock['D']) & (stock['K'].shift(1) >= stock['D'].shift(1)), -1, stock['Signal'])
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

    # 繪製KDJ指標
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['K'], mode='lines', name='K'))
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['D'], mode='lines', name='D'))
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['J'], mode='lines', name='J'))

    fig.update_layout(title=f'KDJ 策略', xaxis_title='日期', yaxis_title='價格', showlegend=True)
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

    # 讀取股票數據
    stock = load_stock_data(stockname, start_date, end_date, interval)
    if stock is not None:
        st.subheader(f"股票代號: {stockname}")
        st.write(stock.head())

        # 計算KDJ指標並繪圖
        stock = calculate_kdj(stock)
        stock = kdj_strategy(stock)
        plot_stock_data(stock, "KDJ")

        # 顯示績效指標
        performance_metrics = calculate_performance(stock)
        st.subheader("績效指標")
        st.write(performance_metrics)

if __name__ == "__main__":
    main()
