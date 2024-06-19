import yfinance as yf
import pandas as pd
import streamlit as st

# 计算KDJ指标
def calculate_kdj(stock):
    low_min = stock['Low'].rolling(window=9).min()
    high_max = stock['High'].rolling(window=9).max()

    rsv = (stock['Close'] - low_min) / (high_max - low_min) * 100
    stock['K'] = rsv.ewm(com=2).mean()
    stock['D'] = stock['K'].ewm(com=2).mean()
    stock['J'] = 3 * stock['K'] - 2 * stock['D']
    return stock

# 计算布林带指标
def calculate_bollinger_bands(stock, window_size=20, num_std_dev=2):
    rolling_mean = stock['Close'].rolling(window=window_size).mean()
    rolling_std = stock['Close'].rolling(window=window_size).std()
    
    stock['Middle_Band'] = rolling_mean
    stock['Upper_Band'] = rolling_mean + (rolling_std * num_std_dev)
    stock['Lower_Band'] = rolling_mean - (rolling_std * num_std_dev)
    
    return stock

# 计算MACD指标
def calculate_macd(stock, short_window=12, long_window=26, signal_window=9):
    short_ema = stock['Close'].ewm(span=short_window, min_periods=1).mean()
    long_ema = stock['Close'].ewm(span=long_window, min_periods=1).mean()
    
    stock['MACD'] = short_ema - long_ema
    stock['Signal_Line'] = stock['MACD'].ewm(span=signal_window, min_periods=1).mean()
    stock['MACD_Histogram'] = stock['MACD'] - stock['Signal_Line']
    
    return stock

# 主函数
def main():
    st.title('股票分析')
    
    # 获取用户输入
    stock_name = st.sidebar.text_input('请输入股票代码（例如AAPL）', 'AAPL')
    strategy_name = st.sidebar.selectbox('请选择策略', ['KDJ', 'Bollinger Bands', 'MACD'])
    
    # 下载股票数据
    stock = yf.download(stock_name, start='2020-01-01', end='2023-01-01')
    
    # 根据选择的策略计算指标
    if strategy_name == 'KDJ':
        stock = calculate_kdj(stock)
    elif strategy_name == 'Bollinger Bands':
        stock = calculate_bollinger_bands(stock)
    elif strategy_name == 'MACD':
        # 注意：MACD需要设置合适的参数
        short_window = st.sidebar.number_input('请输入MACD短期EMA周期', min_value=1, value=12)
        long_window = st.sidebar.number_input('请输入MACD长期EMA周期', min_value=1, value=26)
        signal_window = st.sidebar.number_input('请输入MACD信号线周期', min_value=1, value=9)
        stock = calculate_macd(stock, short_window, long_window, signal_window)
    
    # 显示股票数据和指标
    st.write(stock)

if __name__ == "__main__":
    main()
