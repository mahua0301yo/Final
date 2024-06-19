import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 定義技術指標計算函數
def calculate_indicators(df):
    # 布林通道
    df['Middle_Band'] = df['Close'].rolling(window=20).mean()
    df['Upper_Band'] = df['Middle_Band'] + 1.96*df['Close'].rolling(window=20).std()
    df['Lower_Band'] = df['Middle_Band'] - 1.96*df['Close'].rolling(window=20).std()
    
    # KDJ
    low_list = df['Low'].rolling(9).min()
    low_list.fillna(value=df['Low'].expanding().min(), inplace=True)
    high_list = df['High'].rolling(9).max()
    high_list.fillna(value=df['High'].expanding().max(), inplace=True)
    rsv = (df['Close'] - low_list) / (high_list - low_list) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    
    # 唐奇安通道
    df['Donchian_High'] = df['High'].rolling(window=20).max()
    df['Donchian_Low'] = df['Low'].rolling(window=20).min()

    return df

# 繪圖函數
def plot_bollinger_bands(stock):
    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Candlestick(x=stock['Date'], open=stock['Open'], high=stock['High'], low=stock['Low'], close=stock['Close'], name='K線圖'))
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Middle_Band'], mode='lines', name='中軌'))
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Upper_Band'], mode='lines', name='上軌'))
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Lower_Band'], mode='lines', name='下軌'))

    fig.update_xaxes(title_text='日期')
    fig.update_yaxes(title_text='價格')

    fig.update_layout(title='布林通道策略圖', height=600, showlegend=True)

    st.plotly_chart(fig)

def plot_kdj(stock):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    fig.add_trace(go.Candlestick(x=stock['Date'], open=stock['Open'], high=stock['High'], low=stock['Low'], close=stock['Close'], name='K線圖'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['K'], mode='lines', name='K值'), row=2, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['D'], mode='lines', name='D值'), row=2, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['J'], mode='lines', name='J值'), row=2, col=1)

    # 隱藏中間的時間軸標籤
    fig.update_xaxes(title_text='日期', row=2, col=1, showticklabels=True)
    fig.update_xaxes(showticklabels=False, row=1, col=1)

    fig.update_yaxes(title_text='價格', row=1, col=1)
    fig.update_yaxes(title_text='KDJ值', row=2, col=1)

    fig.update_layout(title='KDJ策略圖', height=600, showlegend=True)

    st.plotly_chart(fig)

def plot_rsi(stock):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.5, 0.2])

    fig.add_trace(go.Candlestick(x=stock['Date'], open=stock['Open'], high=stock['High'], low=stock['Low'], close=stock['Close'], name='K線圖'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['RSI'], mode='lines', name='RSI'), row=2, col=1)

    fig.add_trace(go.Scatter(x=stock['Date'], y=[80]*len(stock), mode='lines', name='超買', line=dict(color='rgba(255, 0, 0, 0.5)', dash='dash')), row=2, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=[20]*len(stock), mode='lines', name='超賣', line=dict(color='rgba(0, 0, 255, 0.5)', dash='dash')), row=2, col=1)

    # 隱藏中間的時間軸標籤
    fig.update_xaxes(title_text='日期', row=2, col=1, showticklabels=True)
    fig.update_xaxes(showticklabels=False, row=1, col=1)

    fig.update_yaxes(title_text='價格', row=1, col=1)
    fig.update_yaxes(title_text='RSI值', row=2, col=1)

    fig.update_layout(title='RSI策略圖', height=600, showlegend=True)

    st.plotly_chart(fig)

def plot_macd(stock):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.5, 0.2])

    fig.add_trace(go.Candlestick(x=stock['Date'], open=stock['Open'], high=stock['High'], low=stock['Low'], close=stock['Close'], name='K線圖'), row=1, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['MACD'], mode='lines', name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Signal_Line'], mode='lines', name='信號線'), row=2, col=1)
    fig.add_trace(go.Bar(x=stock['Date'], y=stock['MACD_Hist'], name='柱狀圖'), row=2, col=1)

    # 隱藏中間的時間軸標籤
    fig.update_xaxes(title_text='日期', row=2, col=1, showticklabels=True)
    fig.update_xaxes(showticklabels=False, row=1, col=1)

    fig.update_yaxes(title_text='價格', row=1, col=1)
    fig.update_yaxes(title_text='MACD值', row=2, col=1)

    fig.update_layout(title='MACD策略圖', height=600, showlegend=True)

    st.plotly_chart(fig)

def plot_donchian_channels(stock):
    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Candlestick(x=stock['Date'], open=stock['Open'], high=stock['High'], low=stock['Low'], close=stock['Close'], name='K線圖'))
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Donchian_High'], mode='lines', name='唐奇安上軌'))
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Donchian_Low'], mode='lines', name='唐奇安下軌'))

    fig.update_xaxes(title_text='日期')
    fig.update_yaxes(title_text='價格')

    fig.update_layout(title='唐奇安通道策略圖', height=600, showlegend=True)

    st.plotly_chart(fig)

# Streamlit 應用程式介面
st.title('技術指標分析')

# 輸入股票代碼
stock_code = st.text_input('輸入股票代碼', 'AAPL')

# 下載數據
data = yf.download(stock_code, start='2020-01-01', end='2023-01-01')
data.reset_index(inplace=True)

# 計算技術指標
stock_data = calculate_indicators(data)

# 選擇指標
indicator = st.selectbox('選擇指標', ('布林通道', 'KDJ', 'RSI', 'MACD', '唐奇安通道'))

# 繪製相應的圖表
if indicator == '布林通道':
    plot_bollinger_bands(stock_data)
elif indicator == 'KDJ':
    plot_kdj(stock_data)
elif indicator == 'RSI':
    plot_rsi(stock_data)
elif indicator == 'MACD':
    plot_macd(stock_data)
elif indicator == '唐奇安通道':
    plot_donchian_channels(stock_data)

# 定義交易策略
def trading_strategy(stock, strategy_name):
    # 根據不同的策略進行交易
    if strategy_name == "Bollinger Bands":
        stock['Position'] = np.where(stock['Close'] > stock['Upper_Band'], -1, np.nan)
        stock['Position'] = np.where(stock['Close'] < stock['Lower_Band'], 1, stock['Position'])
    elif strategy_name == "KDJ":
        stock['Position'] = np.where(stock['K'] > stock['D'], 1, -1)
    elif strategy_name == "RSI":
        stock['Position'] = np.where(stock['RSI'] < 20, 1, np.nan)
        stock['Position'] = np.where(stock['RSI'] > 80, -1, stock['Position'])
    elif strategy_name == "MACD":
        stock['Position'] = np.where(stock['MACD'] > stock['Signal_Line'], 1, -1)
    elif strategy_name == "唐奇安通道":
        stock['Position'] = np.where(stock['Close'] > stock['Donchian_High'].shift(1), 1, np.nan)
        stock['Position'] = np.where(stock['Close'] < stock['Donchian_Low'].shift(1), -1, stock['Position'])

    stock['Position'].fillna(method='ffill', inplace=True)
    stock['Position'].fillna(0, inplace=True)
    stock['Market_Return'] = stock['Close'].pct_change()
    stock['Strategy_Return'] = stock['Market_Return'] * stock['Position'].shift(1)
    stock['Cumulative_Strategy_Return'] = (1 + stock['Strategy_Return']).cumprod() - 1

    # 計算績效指標
    total_trades = len(stock[(stock['Position'] == 1) | (stock['Position'] == -1)])
    winning_trades = len(stock[(stock['Position'].shift(1) == 1) & (stock['Strategy_Return'] > 0)])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    stock['Drawdown'] = (stock['Cumulative_Strategy_Return'].cummax() - stock['Cumulative_Strategy_Return'])
    max_drawdown = stock['Drawdown'].max()

    total_profit = stock['Strategy_Return'].sum()
    consecutive_losses = (stock['Strategy_Return'] < 0).astype(int).groupby(stock['Strategy_Return'].ge(0).cumsum()).sum().max()

    st.write(f"策略績效指標:")
    st.write(f"勝率: {win_rate:.2%}")
    st.write(f"最大連續虧損: {consecutive_losses}")
    st.write(f"最大資金回落: {max_drawdown:.2%}")
    st.write(f"總損益: {total_profit:.2%}")

    return stock

# 主函數
def main():
    st.title("股票技術指標交易策略")

    # 選擇資料區間
    st.sidebar.subheader("選擇資料區間")
    start_date = st.sidebar.date_input('選擇開始日期', datetime.date(2020, 1, 1))
    end_date = st.sidebar.date_input('選擇結束日期', datetime.date(2023, 1, 1))
    stockname = st.sidebar.text_input('請輸入股票代號 (例: 2330.TW)', '2330.TW')

    # 選擇K線時間長
    interval_options = {"1天": "1d", "1星期": "1wk", "1個月": "1mo"}
    interval_label = st.sidebar.selectbox("選擇K線時間長", list(interval_options.keys()))
    interval = interval_options[interval_label]

    # 選擇指標和參數
    strategy_name = st.sidebar.selectbox("選擇指標", ["Bollinger Bands", "KDJ", "RSI", "MACD", "唐奇安通道"])

    if strategy_name == "Bollinger Bands":
        bollinger_period = st.sidebar.slider("布林通道週期", min_value=5, max_value=50, value=20, step=1)
        bollinger_std = st.sidebar.slider("布林通道標準差倍數", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    elif strategy_name == "KDJ":
        kdj_period = st.sidebar.slider("KDJ週期", min_value=5, max_value=50, value=14, step=1)
    elif strategy_name == "RSI":
        rsi_period = st.sidebar.slider("RSI週期", min_value=5, max_value=50, value=14, step=1)
    elif strategy_name == "MACD":
        short_window = st.sidebar.slider("短期EMA窗口", min_value=5, max_value=50, value=12, step=1)
        long_window = st.sidebar.slider("長期EMA窗口", min_value=10, max_value=100, value=26, step=1)
        signal_window = st.sidebar.slider("信號線窗口", min_value=5, max_value=50, value=9, step=1)
    elif strategy_name == "唐奇安通道":
        donchian_period = st.sidebar.slider("唐奇安通道週期", min_value=5, max_value=50, value=20, step=1)

    stock = load_stock_data(stockname, start_date, end_date, interval)
    if stock is not None:
        st.subheader(f"股票代號: {stockname}")
        st.write(stock.head())

        if strategy_name == "Bollinger Bands":
            stock = calculate_bollinger_bands(stock, period=bollinger_period, std_dev=bollinger_std)
            plot_bollinger_bands(stock)
        elif strategy_name == "KDJ":
            stock = calculate_kdj(stock, period=kdj_period)
            plot_kdj(stock)
        elif strategy_name == "RSI":
            stock = calculate_rsi(stock, period=rsi_period)
            plot_rsi(stock)
        elif strategy_name == "MACD":
            stock = calculate_macd(stock, short_window=short_window, long_window=long_window, signal_window=signal_window)
            plot_macd(stock)
        elif strategy_name == "唐奇安通道":
            stock = calculate_donchian_channels(stock, period=donchian_period)
            plot_donchian_channels(stock)
        
        stock = trading_strategy(stock, strategy_name)
        st.write(f"策略績效 (累積報酬): {stock['Cumulative_Strategy_Return'].iloc[-1]:.2%}")

if __name__ == "__main__":
    main()
