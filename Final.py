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
    
    fig_macd.update_layout(title='MACD指標',
                           xaxis_title='日期',
                           yaxis_title='數值')
    st.plotly_chart(fig_macd)

# 繪製唐奇安通道指標
def plot_donchian_channels(stock):
    plot_stock_data(stock, "唐奇安通道")

    fig_donchian = go.Figure()
    fig_donchian.add_trace(go.Scatter(x=stock['Date'], y=stock['Upper_Channel'], line=dict(color='green', width=1), name='唐奇安通道上通道'))
    fig_donchian.add_trace(go.Scatter(x=stock['Date'], y=stock['Lower_Channel'], line=dict(color='green', width=1), name='唐奇安通道下通道'))
    
    fig_donchian.update_layout(title='唐奇安通道指標',
                               xaxis_title='日期',
                               yaxis_title='價格')
    st.plotly_chart(fig_donchian)


# 回測各指標的策略
def backtest_strategy(stock, strategy_name):
    if strategy_name == '布林通道':
        stock = calculate_bollinger_bands(stock)
        plot_stock_data(stock, strategy_name)
    elif strategy_name == 'KDJ':
        stock = calculate_kdj(stock)
        plot_kdj(stock)
    elif strategy_name == 'RSI':
        stock = calculate_rsi(stock)
        plot_rsi(stock)
    elif strategy_name == 'MACD':
        stock = calculate_macd(stock)
        plot_macd(stock)
    elif strategy_name == '唐奇安通道':
        stock = calculate_donchian_channels(stock)
        plot_donchian_channels(stock)
    else:
        st.error("未知的策略名稱")


# 主程式碼
def main():
    st.title('股票交易策略回測')
    
    # 輸入股票代號和日期範圍
    stockname = st.text_input('輸入股票代號 (例如AAPL):')
    start_date = st.date_input('選擇開始日期:')
    end_date = st.date_input('選擇結束日期:')
    
    # 選擇回測策略
    strategies = ['布林通道', 'KDJ', 'RSI', 'MACD', '唐奇安通道']
    strategy_name = st.selectbox('選擇交易策略:', strategies)
    
    if st.button('開始回測'):
        # 讀取股票數據
        stock = load_stock_data(stockname, start_date, end_date, '1d')
        
        if stock is not None:
            # 進行回測
            backtest_strategy(stock, strategy_name)


if __name__ == '__main__':
    main()
