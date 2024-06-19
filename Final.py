# 載入必要模組
import yfinance as yf
import pandas as pd
import streamlit as st
import numpy as np
import datetime
import mplfinance as mpf

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
    stock['EMA_short'] = stock['Close'].ewm(span=short_window, adjust=False).mean()
    stock['EMA_long'] = stock['Close'].ewm(span=long_window, adjust=False).mean()
    stock['MACD'] = stock['EMA_short'] - stock['EMA_long']
    stock['Signal_Line'] = stock['MACD'].ewm(span=signal_window, adjust=False).mean()
    stock['Histogram'] = stock['MACD'] - stock['Signal_Line']
    return stock

# 定義函數來計算唐奇安通道指標
def calculate_donchian_channels(stock, period=20):
    stock['Donchian_High'] = stock['High'].rolling(window=period).max()
    stock['Donchian_Low'] = stock['Low'].rolling(window=period).min()
    return stock

# 定義單獨的繪圖函數
def plot_bollinger_bands(stock):
    add_plot = [
        mpf.make_addplot(stock['Middle_Band'], color='orange'),
        mpf.make_addplot(stock['Upper_Band'], color='b'),
        mpf.make_addplot(stock['Lower_Band'], color='b'),
        mpf.make_addplot(stock['Volume'], panel=1, color='purple', ylabel='Volume')
    ]
    mpf.plot(stock, type='candle', addplot=add_plot, title="布林通道策略圖", ylabel='Price')

def plot_kdj(stock):
    add_plot = [
        mpf.make_addplot(stock['K'], color='purple'),
        mpf.make_addplot(stock['D'], color='orange'),
        mpf.make_addplot(stock['J'], color='blue')
    ]
    mpf.plot(stock, type='candle', addplot=add_plot, title="KDJ策略圖", ylabel='Price')

def plot_rsi(stock):
    add_plot = [
        mpf.make_addplot(stock['RSI'], color='purple')
    ]
    mpf.plot(stock, type='candle', addplot=add_plot, title="RSI策略圖", ylabel='Price')

def plot_macd(stock):
    add_plot = [
        mpf.make_addplot(stock['MACD'], color='purple'),
        mpf.make_addplot(stock['Signal_Line'], color='orange'),
        mpf.make_addplot(stock['Histogram'], type='bar', color='grey', ylabel='Histogram')
    ]
    mpf.plot(stock, type='candle', addplot=add_plot, title="MACD策略圖", ylabel='Price')

def plot_donchian_channels(stock):
    add_plot = [
        mpf.make_addplot(stock['Donchian_High'], color='orange'),
        mpf.make_addplot(stock['Donchian_Low'], color='blue'),
        mpf.make_addplot(stock['Volume'], panel=1, color='purple', ylabel='Volume')
    ]
    mpf.plot(stock, type='candle', addplot=add_plot, title="唐奇安通道策略圖", ylabel='Price')

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
    st.write(f"最大資金回撤: {max_drawdown:.2%}")
    st.write(f"總收益率: {total_profit:.2%}")

    # 繪製策略表現圖
    fig, ax = mpf.plot(stock, type='line', addplot=[mpf.make_addplot(stock['Cumulative_Strategy_Return'], ylabel='Cumulative Return')], 
                       title=f"{strategy_name} 策略表現", ylabel='Cumulative Return', returnfig=True)
    st.pyplot(fig)

def main():
    # Streamlit App 的主要程式碼
    st.title('股票技術指標分析與交易策略')
    
    # 設定選擇股票及日期區間
    stockname = st.sidebar.text_input("請輸入股票代號（例如：AAPL）")
    start_date = st.sidebar.date_input("請選擇開始日期")
    end_date = st.sidebar.date_input("請選擇結束日期", datetime.date.today())
    interval = st.sidebar.selectbox("請選擇數據間隔", ["1d", "1wk", "1mo"])

    # 讀取股票數據
    if stockname:
        stock = load_stock_data(stockname, start_date, end_date, interval)
        if stock is not None:
            st.write(f"顯示 {stockname} 的股票數據:")
            st.dataframe(stock.head())

            # 計算技術指標
            stock = calculate_bollinger_bands(stock)
            stock = calculate_kdj(stock)
            stock = calculate_rsi(stock)
            stock = calculate_macd(stock)
            stock = calculate_donchian_channels(stock)

            # 選擇繪製的技術指標圖表
            st.subheader("選擇繪製的技術指標圖表:")
            plot_option = st.selectbox("請選擇技術指標", ["布林通道策略圖", "KDJ策略圖", "RSI策略圖", "MACD策略圖", "唐奇安通道策略圖"])

            if plot_option == "布林通道策略圖":
                plot_bollinger_bands(stock)
            elif plot_option == "KDJ策略圖":
                plot_kdj(stock)
            elif plot_option == "RSI策略圖":
                plot_rsi(stock)
            elif plot_option == "MACD策略圖":
                plot_macd(stock)
            elif plot_option == "唐奇安通道策略圖":
                plot_donchian_channels(stock)

            # 選擇交易策略
            st.subheader("選擇交易策略並評估其績效:")
            strategy_option = st.selectbox("請選擇交易策略", ["Bollinger Bands", "KDJ", "RSI", "MACD", "唐奇安通道"])

            # 執行交易策略並顯示績效
            if st.button("執行交易策略"):
                trading_strategy(stock, strategy_option)

if __name__ == "__main__":
    main()
