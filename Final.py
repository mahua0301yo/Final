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

# 計算策略回測績效
def calculate_strategy_performance(stock, buy_signal_col, sell_signal_col):
    # 初始化交易訊號為0
    stock['signal'] = 0
    
    # 買入訊號為1，賣出訊號為-1
    stock.loc[stock[buy_signal_col] == 1, 'signal'] = 1
    stock.loc[stock[sell_signal_col] == 1, 'signal'] = -1
    
    # 計算每日收益率
    stock['Returns'] = stock['Close'].pct_change()
    stock['Strategy_Returns'] = stock['Returns'] * stock['signal'].shift(1)
    
    # 計算持有期間
    if isinstance(stock.index, pd.DatetimeIndex):  # 只有在日期時間索引時才使用.dt.days
        stock['Trade_Duration'] = stock.index.to_series().diff().dt.days
    else:
        stock['Trade_Duration'] = stock.index.to_series().diff().apply(lambda x: x.days if pd.notnull(x) else 0)
    
    # 計算勝率
    total_trades = stock[buy_signal_col].sum() + stock[sell_signal_col].sum()
    win_trades = (stock['Strategy_Returns'] > 0).sum()
    win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0
    
    return stock, win_rate

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
    fig_rsi.add_trace(go.Scatter(x=stock['Date'], y=[70]*len(stock), line=dict(color='red', width=1), name='Overbought', line_dash='dash'))
    fig_rsi.add_trace(go.Scatter(x=stock['Date'], y=[30]*len(stock), line=dict(color='green', width=1), name='Oversold', line_dash='dash'))

    fig_rsi.update_layout(title='RSI指標',
                          xaxis_title='日期',
                          yaxis_title='數值')
    st.plotly_chart(fig_rsi)

# 繪製MACD指標
def plot_macd(stock):
    plot_stock_data(stock, "MACD")

    fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             vertical_spacing=0.02, row_heights=[0.7, 0.3])
    
    fig_macd.add_trace(go.Scatter(x=stock['Date'], y=stock['MACD'], line=dict(color='blue', width=1), name='MACD'), row=1, col=1)
    fig_macd.add_trace(go.Scatter(x=stock['Date'], y=stock['Signal_Line'], line=dict(color='red', width=1), name='Signal Line'), row=1, col=1)

    fig_macd.add_trace(go.Bar(x=stock['Date'], y=stock['MACD_Histogram'], marker_color='gray', name='Histogram'), row=2, col=1)

    fig_macd.update_layout(title='MACD指標',
                           xaxis_title='日期',
                           yaxis_title='數值')
    st.plotly_chart(fig_macd)

# 繪製唐奇安通道指標
def plot_donchian_channels(stock):
    plot_stock_data(stock, "唐奇安通道")

    fig_donchian = go.Figure()
    fig_donchian.add_trace(go.Scatter(x=stock['Date'], y=stock['Upper_Channel'], line=dict(color='green', width=1), name='上通道'))
    fig_donchian.add_trace(go.Scatter(x=stock['Date'], y=stock['Lower_Channel'], line=dict(color='red', width=1), name='下通道'))
    
    fig_donchian.update_layout(title='唐奇安通道指標',
                               xaxis_title='日期',
                               yaxis_title='價格')
    st.plotly_chart(fig_donchian)

# 主程式碼
def main():
    # 設置網頁標題和介紹
    st.title('股票分析')
    st.write('這是一個用於股票技術指標分析和策略回測的應用。')
    
    # 使用者輸入股票代碼和日期範圍
    stockname = st.text_input('輸入股票代碼（例如AAPL）：', 'AAPL')
    start_date = st.text_input('輸入開始日期（YYYY-MM-DD）：', '2020-01-01')
    end_date = st.text_input('輸入結束日期（YYYY-MM-DD）：', '2021-01-01')
    interval = st.selectbox('選擇股票數據的間隔', ('1d', '1wk', '1mo'), index=0)
    
    # 載入股票數據
    stock = load_stock_data(stockname, start_date, end_date, interval)
    if stock is None:
        return
    
    # 選擇要分析的策略
    strategy_name = st.selectbox('選擇分析的策略', ('布林通道', 'KDJ', 'RSI', 'MACD', '唐奇安通道'))
    
    # 計算所選策略的指標
    if strategy_name == '布林通道':
        stock = calculate_bollinger_bands(stock)
        signals, win_rate = bollinger_band_strategy(stock, period=20)
        st.write("策略回測績效:")
        st.write(f"勝率: {win_rate:.2f}%")
        st.write(signals.head())
        plot_stock_data(signals, strategy_name)

    elif strategy_name == 'KDJ':
        stock = calculate_kdj(stock)
        plot_kdj(stock)

    elif strategy_name == 'RSI':
        stock = calculate_rsi(stock)
        signals, win_rate = rsi_strategy(stock)
        st.write("策略回測績效:")
        st.write(f"勝率: {win_rate:.2f}%")
        st.write(signals.head())
        plot_rsi(signals)

    elif strategy_name == 'MACD':
        stock = calculate_macd(stock)
        signals, win_rate = macd_strategy(stock)
        st.write("策略回測績效:")
        st.write(f"勝率: {win_rate:.2f}%")
        st.write(signals.head())
        plot_macd(signals)

    elif strategy_name == '唐奇安通道':
        stock = calculate_donchian_channels(stock)
        signals, win_rate = donchian_channel_strategy(stock, period=20)
        st.write("策略回測績效:")
        st.write(f"勝率: {win_rate:.2f}%")
        st.write(signals.head())
        plot_donchian_channels(signals)

if __name__ == '__main__':
    main()
