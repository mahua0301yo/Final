# 載入必要模組
import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots

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
    delta = stock['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    stock['RSI'] = 100 - (100 / (1 + rs))
    
    # 新增超買和超賣標記
    stock['Overbought'] = stock['RSI'] > 80
    stock['Oversold'] = stock['RSI'] < 20
    
    return stock

# 定義函數來計算MACD指標
def calculate_macd(stock, short_window=12, long_window=26, signal_window=9):
    stock['EMA12'] = stock['Close'].ewm(span=short_window, adjust=False).mean()
    stock['EMA26'] = stock['Close'].ewm(span=long_window, adjust=False).mean()
    stock['MACD'] = stock['EMA12'] - stock['EMA26']
    stock['Signal_Line'] = stock['MACD'].ewm(span=signal_window, adjust=False).mean()
    stock.drop(columns=['EMA12', 'EMA26'], inplace=True)
    return stock

# 定義函數來計算唐奇安通道指標
def calculate_donchian_channels(stock, period=20):
    stock['Upper_Channel'] = stock['Close'].rolling(window=period).max()
    stock['Lower_Channel'] = stock['Close'].rolling(window=period).min()
    return stock

# 繪製股票數據和指標圖
def plot_stock_data(stock, strategy_name):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02, row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(x=stock['Date'],
                                 open=stock['Open'],
                                 high=stock['High'],
                                 low=stock['Low'],
                                 close=stock['Close'],
                                 name='價格'), row=1, col=1)

    if 'Upper_Band' in stock.columns:
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Upper_Band'], line=dict(color='red', width=1), name='上軌'), row=1, col=1)
    if 'Middle_Band' in stock.columns:
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Middle_Band'], line=dict(color='blue', width=1), name='中軌'), row=1, col=1)
    if 'Lower_Band' in stock.columns:
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Lower_Band'], line=dict(color='red', width=1), name='下軌'), row=1, col=1)

    # 加入唐奇安通道（使用實線）
    if 'Upper_Channel' in stock.columns:
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Upper_Channel'], line=dict(color='green', width=1), name='唐奇安通道上通道'), row=1, col=1)
    if 'Lower_Channel' in stock.columns:
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Lower_Channel'], line=dict(color='green', width=1), name='唐奇安通道下通道'), row=1, col=1)
    
    fig.add_trace(go.Bar(x=stock['Date'], y=stock['amount'], name='交易量'), row=2, col=1)

    fig.update_layout(title=f"{strategy_name}策略 - 股票價格與交易量",
                      xaxis_title='日期',
                      yaxis_title='價格')

    st.plotly_chart(fig)

# 繪製KDJ指標
def plot_kdj(stock):
    plot_stock_data(stock, "KDJ")

    fig_kdj = go.Figure()
    fig_kdj.add_trace(go.Scatter(x=stock['Date'], y=stock['K'], line=dict(color='blue', width=1), name='K'))
    fig_kdj.add_trace(go.Scatter(x=stock['Date'], y=stock['D'], line=dict(color='orange', width=1), name='D'))
    fig_kdj.add_trace(go.Scatter(x=stock['Date'], y=stock['J'], line=dict(color='green', width=1), name='J'))
    
    fig_kdj.update_layout(title='KDJ指標',
                          xaxis_title='日期',
                          yaxis_title='數值')
    st.plotly_chart(fig_kdj)

# 繪製RSI指標
def plot_rsi(stock):
    plot_stock_data(stock, "RSI")

    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=stock['Date'], y=stock['RSI'], line=dict(color='purple', width=1), name='RSI'))
    
    # 加入超買區和超賣區的背景色
    fig_rsi.add_shape(type="rect", xref="paper", yref="y",
                      x0=0, y0=80, x1=1, y1=100,
                      fillcolor="rgba(255, 0, 0, 0.3)",  # 紅色半透明填充
                      layer="below", line_width=0,
                      name="超買區域")
    
    fig_rsi.add_shape(type="rect", xref="paper", yref="y",
                      x0=0, y0=0, x1=1, y1=20,
                      fillcolor="rgba(0, 0, 255, 0.3)",  # 藍色半透明填充
                      layer="below", line_width=0,
                      name="超賣區域")
    
    fig_rsi.update_layout(title='RSI指標',
                          xaxis_title='日期',
                          yaxis_title='數值')
    st.plotly_chart(fig_rsi)


# 繪製MACD指標
def plot_macd(stock):
    plot_stock_data(stock, "MACD")

    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=stock['Date'], y=stock['MACD'], line=dict(color='blue', width=1), name='MACD'))
    fig_macd.add_trace(go.Scatter(x=stock['Date'], y=stock['Signal_Line'], line=dict(color='red', width=1), name='Signal Line'))

    fig_macd.add_trace(go.Bar(x=stock['Date'], y=stock['MACD'] - stock['Signal_Line'], marker_color='grey', name='MACD Histogram'))

    fig_macd.update_layout(title='MACD指標',
                           xaxis_title='日期',
                           yaxis_title='數值')
    st.plotly_chart(fig_macd)

# Streamlit應用程式主體
def main():
    st.title("股票技術分析工具")
    
    stockname = st.sidebar.text_input("輸入股票代號", value='AAPL')
    start_date = st.sidebar.date_input("選擇開始日期", value=pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("選擇結束日期", value=pd.to_datetime("2023-12-31"))
    interval = st.sidebar.selectbox("選擇數據頻率", options=['1d', '1wk', '1mo'], index=0)
    strategy_name = st.sidebar.selectbox("選擇交易策略", options=["Bollinger Bands", "KDJ", "RSI", "MACD", "唐奇安通道"], index=0)

    # 初始化交易策略參數
    long_ma_period = 50  # 長期移動平均週期
    short_ma_period = 20  # 短期移動平均週期
    move_stop_loss = 0.05  # 移動停損

    # 載入股票資料
    stock = load_stock_data(stockname, start_date, end_date, interval)
    if stock is not None:
        st.subheader(f"股票代號: {stockname}")
        st.write(stock.head())

        # 根據選擇的策略進行分析並繪製相應的圖表
        if strategy_name == "Bollinger Bands":
            bollinger_period = st.sidebar.slider("布林通道週期", min_value=5, max_value=50, value=20, step=1)
            bollinger_std = st.sidebar.slider("布林通道標準差倍數", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
            stock = calculate_bollinger_bands(stock, period=bollinger_period, std_dev=bollinger_std)
            plot_stock_data(stock, strategy_name)
            # 計算並顯示交易績效
            st.subheader("交易績效 - 布林通道策略")
            trade_record, profit, total_profit, win_rate, acc_loss, mdd = calculate_bollinger_performance(stock)
            st.write(f"交易紀錄: {trade_record}")
            st.write(f"損益: {profit}")
            st.write(f"總損益: {total_profit}")
            st.write(f"勝率: {win_rate * 100:.2f}%")
            st.write(f"最大連續虧損: {acc_loss}")
            st.write(f"最大資金回落 (MDD): {mdd}")
        
        elif strategy_name == "KDJ":
            kdj_period = st.sidebar.slider("KDJ週期", min_value=5, max_value=50, value=14, step=1)
            stock = calculate_kdj(stock, period=kdj_period)
            plot_kdj(stock)
            # 計算並顯示交易績效
            st.subheader("交易績效 - KDJ策略")
            trade_record, profit, total_profit, win_rate, acc_loss, mdd = calculate_kdj_performance(stock)
            st.write(f"交易紀錄: {trade_record}")
            st.write(f"損益: {profit}")
            st.write(f"總損益: {total_profit}")
            st.write(f"勝率: {win_rate * 100:.2f}%")
            st.write(f"最大連續虧損: {acc_loss}")
            st.write(f"最大資金回落 (MDD): {mdd}")
        
        elif strategy_name == "RSI":
            rsi_period = st.sidebar.slider("RSI週期", min_value=5, max_value=50, value=14, step=1)
            stock = calculate_rsi(stock, period=rsi_period)
            plot_rsi(stock)
            # 計算並顯示交易績效
            st.subheader("交易績效 - RSI策略")
            trade_record, profit, total_profit, win_rate, acc_loss, mdd = calculate_rsi_performance(stock)
            st.write(f"交易紀錄: {trade_record}")
            st.write(f"損益: {profit}")
            st.write(f"總損益: {total_profit}")
            st.write(f"勝率: {win_rate * 100:.2f}%")
            st.write(f"最大連續虧損: {acc_loss}")
            st.write(f"最大資金回落 (MDD): {mdd}")
        
        elif strategy_name == "MACD":
            short_window = st.sidebar.slider("短期EMA窗口", min_value=5, max_value=50, value=12, step=1)
            long_window = st.sidebar.slider("長期EMA窗口", min_value=10, max_value=100, value=26, step=1)
            signal_window = st.sidebar.slider("信號線窗口", min_value=5, max_value=50, value=9, step=1)
            stock = calculate_macd(stock, short_window=short_window, long_window=long_window, signal_window=signal_window)
            plot_macd(stock)
            # 計算並顯示交易績效
            st.subheader("交易績效 - MACD策略")
            trade_record, profit, total_profit, win_rate, acc_loss, mdd = calculate_macd_performance(stock)
            st.write(f"交易紀錄: {trade_record}")
            st.write(f"損益: {profit}")
            st.write(f"總損益: {total_profit}")
            st.write(f"勝率: {win_rate * 100:.2f}%")
            st.write(f"最大連續虧損: {acc_loss}")
            st.write(f"最大資金回落 (MDD): {mdd}")
        
        elif strategy_name == "唐奇安通道":
            donchian_period = st.sidebar.slider("唐奇安通道週期", min_value=5, max_value=50, value=20, step=1)
            stock = calculate_donchian_channels(stock, period=donchian_period)
            plot_stock_data(stock, strategy_name)
            # 計算並顯示交易績效
            st.subheader("交易績效 - 唐奇安通道策略")
            trade_record, profit, total_profit, win_rate, acc_loss, mdd = calculate_donchian_performance(stock)
            st.write(f"交易紀錄: {trade_record}")
            st.write(f"損益: {profit}")
            st.write(f"總損益: {total_profit}")
            st.write(f"勝率: {win_rate * 100:.2f}%")
            st.write(f"最大連續虧損: {acc_loss}")
            st.write(f"最大資金回落 (MDD): {mdd}")

# 計算布林通道交易績效的函數
def calculate_bollinger_performance(stock):
    # 假設這裡是計算布林通道策略的交易績效的地方
    trade_record = "布林通道交易紀錄"
    profit = 1200  # 示範損益
    total_profit = 6000  # 示範總損益
    win_rate = 0.68  # 示範勝率
    acc_loss = 4  # 示範最大連續虧損
    mdd = 1000  # 示範最大資金回落
    return trade_record, profit, total_profit, win_rate, acc_loss, mdd

# 計算KDJ交易績效的函數
def calculate_kdj_performance(stock):
    # 假設這裡是計算KDJ策略的交易績效的地方
    trade_record = "KDJ交易紀錄"
    profit = 800  # 示範損益
    total_profit = 5000  # 示範總損益
    win_rate = 0.65  # 示範勝率
    acc_loss = 3  # 示範最大連續虧損
    mdd = 800  # 示範最大資金回落
    return trade_record, profit, total_profit, win_rate, acc_loss, mdd

# 計算RSI交易績效的函數
def calculate_rsi_performance(stock):
    # 假設這裡是計算RSI策略的交易績效的地方
    trade_record = "RSI交易紀錄"
    profit = 1000  # 示範損益
    total_profit = 7000  # 示範總損益
    win_rate = 0.70  # 示範勝率
    acc_loss = 5  # 示範最大連續虧損
    mdd = 1200  # 示範最大資金回落
    return trade_record, profit, total_profit, win_rate, acc_loss, mdd

# 計算MACD交易績效的函數
def calculate_macd_performance(stock):
    # 假設這裡是計算MACD策略的交易績效的地方
    trade_record = "MACD交易紀錄"
    profit = 1500  # 示範損益
    total_profit = 8000  # 示範總損益
    win_rate = 0.72  # 示範勝率
    acc_loss = 3  # 示範最大連續虧損
    mdd = 900  # 示範最大資金回落
    return trade_record, profit, total_profit, win_rate, acc_loss, mdd

# 計算唐奇安通道交易績效的函數
def calculate_donchian_performance(stock):
    # 假設這裡是計算唐奇安通道策略的交易績效的地方
    trade_record = "唐奇安通道交易紀錄"
    profit = 900  # 示範損益
    total_profit = 5500  # 示範總損益
    win_rate = 0.67  # 示範勝率
    acc_loss = 4  # 示範最大連續虧損
    mdd = 950  # 示範最大資金回落
    return trade_record, profit, total_profit, win_rate, acc_loss, mdd

if __name__ == "__main__":
    main()
