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

# 建立交易策略
def trading_strategy(stock, long_ma_period, short_ma_period, move_stop_loss):
    stock['MA_long'] = stock['Close'].rolling(window=long_ma_period).mean()
    stock['MA_short'] = stock['Close'].rolling(window=short_ma_period).mean()

    order_record = []
    order_price = None
    stop_loss_point = None
    capital = 1000000

    for n in range(1, len(stock)):
        if not pd.isnull(stock['MA_long'][n-1]):
            if len(order_record) == 0:
                if stock['MA_short'][n-1] <= stock['MA_long'][n-1] and stock['MA_short'][n] > stock['MA_long'][n]:
                    order_record.append({
                        'action': 'Buy',
                        'date': stock['Date'][n],
                        'price': stock['Open'][n+1],
                        'amount': capital // stock['Open'][n+1]
                    })
                    order_price = stock['Open'][n+1]
                    stop_loss_point = order_price - move_stop_loss
                elif stock['MA_short'][n-1] >= stock['MA_long'][n-1] and stock['MA_short'][n] < stock['MA_long'][n]:
                    order_record.append({
                        'action': 'Sell',
                        'date': stock['Date'][n],
                        'price': stock['Open'][n+1],
                        'amount': capital // stock['Open'][n+1]
                    })
                    order_price = stock['Open'][n+1]
                    stop_loss_point = order_price + move_stop_loss
            elif order_record[-1]['action'] == 'Buy':
                if stock['Close'][n] - move_stop_loss > stop_loss_point:
                    stop_loss_point = stock['Close'][n] - move_stop_loss
                elif stock['Close'][n] < stop_loss_point:
                    order_record.append({
                        'action': 'Sell',
                        'date': stock['Date'][n],
                        'price': stock['Open'][n+1],
                        'amount': order_record[-1]['amount']
                    })
            elif order_record[-1]['action'] == 'Sell':
                if stock['Close'][n] + move_stop_loss < stop_loss_point:
                    stop_loss_point = stock['Close'][n] + move_stop_loss
                elif stock['Close'][n] > stop_loss_point:
                    order_record.append({
                        'action': 'Buy',
                        'date': stock['Date'][n],
                        'price': stock['Open'][n+1],
                        'amount': capital // stock['Open'][n+1]
                    })

    return order_record

# 計算總損益和報酬率
def calculate_performance(order_record, stock):
    total_profit = 0
    initial_capital = 1000000
    
    for order in order_record:
        if order['action'] == 'Buy':
            buy_price = order['price']
            sell_price = stock[stock['Date'] > order['date']]['Open'].iloc[0]
            profit = (sell_price - buy_price) * order['amount']
            total_profit += profit
        elif order['action'] == 'Sell':
            sell_price = order['price']
            buy_price = stock[stock['Date'] > order['date']]['Open'].iloc[0]
            profit = (sell_price - buy_price) * order['amount']
            total_profit += profit
    
    return total_profit, (total_profit / initial_capital) * 100

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

    if 'K' in stock.columns:
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['K'], line=dict(color='blue', width=1), name='K'), row=1, col=1)
    if 'D' in stock.columns:
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['D'], line=dict(color='red', width=1), name='D'), row=1, col=1)

    if 'RSI' in stock.columns:
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['RSI'], line=dict(color='blue', width=1), name='RSI'), row=2, col=1)
        fig.add_trace(go.Scatter(x=stock['Date'], y=[70]*len(stock), line=dict(color='red', width=1, dash='dash'), name='Overbought'), row=2, col=1)
        fig.add_trace(go.Scatter(x=stock['Date'], y=[30]*len(stock), line=dict(color='green', width=1, dash='dash'), name='Oversold'), row=2, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_layout(title=f"{strategy_name} 策略", xaxis_title="日期", yaxis_title="價格")
    st.plotly_chart(fig)

# 主程式
def main():
    stockname = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2021-01-01'
    interval = '1d'
    long_ma_period = 50
    short_ma_period = 20
    move_stop_loss = 2

    # 讀取股票數據
    stock = load_stock_data(stockname, start_date, end_date, interval)
    if stock is None:
        return
    
    # 計算指標
    stock = calculate_bollinger_bands(stock, period=20, std_dev=2)
    stock = calculate_kdj(stock, period=14)
    stock = calculate_rsi(stock, period=14)
    stock = calculate_macd(stock, short_window=12, long_window=26, signal_window=9)
    stock = calculate_donchian_channels(stock, period=20)

    # 打印策略績效
    strategies = {
        'Bollinger Bands': trading_strategy(stock.copy(), long_ma_period, short_ma_period, move_stop_loss),
        'KDJ': trading_strategy(stock.copy(), long_ma_period, short_ma_period, move_stop_loss),
        'RSI': trading_strategy(stock.copy(), long_ma_period, short_ma_period, move_stop_loss),
        'MACD': trading_strategy(stock.copy(), long_ma_period, short_ma_period, move_stop_loss),
        'Donchian Channels': trading_strategy(stock.copy(), long_ma_period, short_ma_period, move_stop_loss)
    }

    for strategy_name, order_record in strategies.items():
        total_profit, return_rate = calculate_performance(order_record, stock)
        st.write(f"{strategy_name} 總損益: {total_profit}")
        st.write(f"{strategy_name} 報酬率: {return_rate:.2f}%")
        plot_stock_data(stock, strategy_name)

# 執行主程式
if __name__ == '__main__':
    main()
