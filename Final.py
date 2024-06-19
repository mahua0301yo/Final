import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np

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

# 定義函數來計算布林通道指標交易策略
def bollinger_band_strategy(stock):
    signals = pd.DataFrame(index=stock.index)
    signals['signal'] = 0.0

    # 生成進出場訊號
    signals['signal'][20:] = np.where(stock['Close'][20:] < stock['Lower_Band'][20:], 1.0, 0.0)
    signals['signal'][20:] = np.where(stock['Close'][20:] > stock['Upper_Band'][20:], -1.0, signals['signal'][20:])

    # 計算每筆交易的回報率
    stock['returns'] = np.log(stock['Close'] / stock['Close'].shift(1))
    signals['strategy_returns'] = signals['signal'].shift(1) * stock['returns']

    return signals

# 定義函數來計算KDJ指標交易策略
def kdj_strategy(stock):
    signals = pd.DataFrame(index=stock.index)
    signals['signal'] = 0.0

    # 生成進出場訊號
    signals['signal'][14:] = np.where(stock['K'][14:] < 20, 1.0, 0.0)
    signals['signal'][14:] = np.where(stock['K'][14:] > 80, -1.0, signals['signal'][14:])

    # 計算每筆交易的回報率
    stock['returns'] = np.log(stock['Close'] / stock['Close'].shift(1))
    signals['strategy_returns'] = signals['signal'].shift(1) * stock['returns']

    return signals

# 定義函數來計算RSI指標交易策略
def rsi_strategy(stock):
    signals = pd.DataFrame(index=stock.index)
    signals['signal'] = 0.0

    # 生成進出場訊號
    signals['signal'][14:] = np.where(stock['RSI'][14:] < 30, 1.0, 0.0)
    signals['signal'][14:] = np.where(stock['RSI'][14:] > 70, -1.0, signals['signal'][14:])

    # 計算每筆交易的回報率
    stock['returns'] = np.log(stock['Close'] / stock['Close'].shift(1))
    signals['strategy_returns'] = signals['signal'].shift(1) * stock['returns']

    return signals

# 定義函數來計算MACD指標交易策略
def macd_strategy(stock):
    signals = pd.DataFrame(index=stock.index)
    signals['signal'] = 0.0

    # 生成進出場訊號
    signals['signal'] = np.where(stock['MACD'] > stock['Signal_Line'], 1.0, 0.0)
    signals['signal'] = np.where(stock['MACD'] < stock['Signal_Line'], -1.0, signals['signal'])

    # 計算每筆交易的回報率
    stock['returns'] = np.log(stock['Close'] / stock['Close'].shift(1))
    signals['strategy_returns'] = signals['signal'].shift(1) * stock['returns']

    return signals

# 定義函數來計算回報率和評估指標
def calculate_returns_and_metrics(signals):
    total_returns = signals['strategy_returns'].cumsum()[-1]
    total_trades = np.sum(signals['signal'] != 0)
    win_trades = np.sum(signals['strategy_returns'] > 0)
    win_rate = win_trades / total_trades if total_trades > 0 else 0.0
    max_drawdown = np.min(signals['strategy_returns'].cumsum() - signals['strategy_returns'].cumsum().expanding().max())

    return total_returns, win_rate, max_drawdown

# 定義函數來繪製股票數據和指標圖
def plot_stock_data(stock, strategy_name, signals=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02, row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(x=stock['Date'],
                                 open=stock['Open'],
                                 high=stock['High'],
                                 low=stock['Low'],
                                 close=stock['Close'],
                                 name='價格'), row=1, col=1)

    if 'Upper_Band' in stock.columns:
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Upper_Band'], line=dict(color='red', width=1), name='上界'), row=1, col=1)
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Middle_Band'], line=dict(color='blue', width=1), name='中界'), row=1, col=1)
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Lower_Band'], line=dict(color='red', width=1), name='下界'), row=1, col=1)

    if signals is not None:
        buy_dates = stock.loc[signals['signal'] == 1.0].index
        sell_dates = stock.loc[signals['signal'] == -1.0].index
        fig.add_trace(go.Scatter(x=buy_dates, y=stock.loc[buy_dates, 'Close'],
                                 mode='markers', marker=dict(color='green', size=8, symbol='triangle-up'),
                                 name='買入信號'), row=1, col=1)
        fig.add_trace(go.Scatter(x=sell_dates, y=stock.loc[sell_dates, 'Close'],
                                 mode='markers', marker=dict(color='red', size=8, symbol='triangle-down'),
                                 name='賣出信號'), row=1, col=1)

    fig.add_trace(go.Bar(x=stock['Date'], y=stock['Volume'], marker=dict(color='blue'), name='成交量'), row=2, col=1)

    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.update_layout(title=f'{strategy_name} 策略交易信號和成交量',
                      xaxis_title='日期',
                      yaxis_title='價格',
                      showlegend=True)

    st.plotly_chart(fig)

# 主函數
def main():
    st.title("股票技術分析工具")
    
    stockname = st.sidebar.text_input("輸入股票代號", value='AAPL')
    start_date = st.sidebar.date_input("選擇開始日期", value=pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("選擇結束日期", value=pd.to_datetime("2023-12-31"))
    interval = st.sidebar.selectbox("選擇數據頻率", options=['1d', '1wk', '1mo'], index=0)
    strategy_name = st.sidebar.selectbox("選擇交易策略", options=["Bollinger Bands", "KDJ", "RSI", "MACD"], index=0)

    stock = load_stock_data(stockname, start_date, end_date, interval)
    if stock is not None:
        st.subheader(f"股票代號: {stockname}")
        st.write(stock.head())

        if strategy_name == "Bollinger Bands":
            stock = calculate_bollinger_bands(stock)
            signals = bollinger_band_strategy(stock)
        elif strategy_name == "KDJ":
            stock = calculate_kdj(stock)
            signals = kdj_strategy(stock)
        elif strategy_name == "RSI":
            stock = calculate_rsi(stock)
            signals = rsi_strategy(stock)
        elif strategy_name == "MACD":
            stock = calculate_macd(stock)
            signals = macd_strategy(stock)

        total_returns, win_rate, max_drawdown = calculate_returns_and_metrics(signals)

        st.subheader(f"{strategy_name} 策略評估指標")
        st.write(f"總回報率: {total_returns:.2f}")
        st.write(f"勝率: {win_rate * 100:.2f}%")
        st.write(f"最大資金回落 (MDD): {max_drawdown:.2f}")

        # 繪製股票數據和指標圖
        plot_stock_data(stock, strategy_name, signals)

if __name__ == "__main__":
    main()
