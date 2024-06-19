import yfinance as yf
import pandas as pd
import streamlit as st
import numpy as np
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
class OrderRecord:
    def __init__(self):
        self.orders = []
        self.profit = 0
        self.total_profit = 0
        self.wins = 0
        self.losses = 0
        self.acc_loss = 0
        self.max_drawdown = 0
        self.open_interest = 0

    def Order(self, action, order_date, cover_date, price, qty):
        self.orders.append({'action': action, 'order_date': order_date, 'cover_date': cover_date, 'price': price, 'qty': qty})
        self.open_interest += qty if action == 'Buy' else -qty

    def Cover(self, action, order_date, cover_date, price, qty):
        for order in self.orders:
            if order['qty'] == qty and ((action == 'Sell' and order['action'] == 'Buy') or (action == 'Buy' and order['action'] == 'Sell')):
                profit = (price - order['price']) * qty if action == 'Sell' else (order['price'] - price) * qty
                self.profit += profit
                self.total_profit += profit
                self.wins += 1 if profit > 0 else 0
                self.losses += 1 if profit <= 0 else 0
                self.acc_loss += profit if profit < 0 else 0
                self.max_drawdown = min(self.max_drawdown, self.acc_loss)
                self.orders.remove(order)
                break
        self.open_interest -= qty if action == 'Sell' else -qty

    def GetOpenInterest(self):
        return self.open_interest

    def GetTradeRecord(self):
        return self.orders

    def GetProfit(self):
        return self.profit

    def GetTotalProfit(self):
        return self.total_profit

    def GetWinRate(self):
        return self.wins / (self.wins + self.losses) if (self.wins + self.losses) > 0 else 0

    def GetAccLoss(self):
        return self.acc_loss

    def GetMDD(self):
        return self.max_drawdown

# 定義函數來計算布林通道指標
def calculate_bollinger_bands(stock, period=20, std_dev=2):
    stock['Middle_Band'] = stock['Close'].rolling(window=period).mean()
    stock['Upper_Band'] = stock['Middle_Band'] + std_dev * stock['Close'].rolling(window=period).std()
    stock['Lower_Band'] = stock['Middle_Band'] - std_dev * stock['Close'].rolling(window=period).std()
    return stock

# 布林通道交易策略
def bollinger_bands_strategy(stock, move_stop_loss):
    stock = calculate_bollinger_bands(stock)
    
    order_record = OrderRecord()
    order_price = None
    stop_loss_point = None

    for n in range(1, len(stock)):
        if not np.isnan(stock['Middle_Band'][n-1]):
            # 無未平倉部位
            if order_record.GetOpenInterest() == 0:
                # 多單進場
                if stock['Close'][n] < stock['Lower_Band'][n]:
                    order_record.Order('Buy', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)
                    order_price = stock['Open'][n+1]
                    stop_loss_point = order_price - move_stop_loss
                # 空單進場
                elif stock['Close'][n] > stock['Upper_Band'][n]:
                    order_record.Order('Sell', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)
                    order_price = stock['Open'][n+1]
                    stop_loss_point = order_price + move_stop_loss
            # 多單出場
            elif order_record.GetOpenInterest() == 1:
                if stock['Close'][n] - move_stop_loss > stop_loss_point:
                    stop_loss_point = stock['Close'][n] - move_stop_loss
                elif stock['Close'][n] < stop_loss_point:
                    order_record.Cover('Sell', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)
            # 空單出場
            elif order_record.GetOpenInterest() == -1:
                if stock['Close'][n] + move_stop_loss < stop_loss_point:
                    stop_loss_point = stock['Close'][n] + move_stop_loss
                elif stock['Close'][n] > stop_loss_point:
                    order_record.Cover('Buy', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)

    return order_record

# KDJ 指標計算
def calculate_kdj(stock, period=14):
    low_min = stock['Low'].rolling(window=period).min()
    high_max = stock['High'].rolling(window=period).max()
    rsv = (stock['Close'] - low_min) / (high_max - low_min) * 100
    stock['K'] = rsv.ewm(com=2).mean()
    stock['D'] = stock['K'].ewm(com=2).mean()
    stock['J'] = 3 * stock['K'] - 2 * stock['D']
    return stock

# KDJ 交易策略
def kdj_strategy(stock, move_stop_loss):
    stock = calculate_kdj(stock)
    
    order_record = OrderRecord()
    order_price = None
    stop_loss_point = None

    for n in range(1, len(stock)):
        if not np.isnan(stock['K'][n-1]):
            # 無未平倉部位
            if order_record.GetOpenInterest() == 0:
                # 多單進場
                if stock['J'][n-1] < stock['K'][n-1] and stock['J'][n] > stock['K'][n]:
                    order_record.Order('Buy', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)
                    order_price = stock['Open'][n+1]
                    stop_loss_point = order_price - move_stop_loss
                # 空單進場
                elif stock['J'][n-1] > stock['K'][n-1] and stock['J'][n] < stock['K'][n]:
                    order_record.Order('Sell', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)
                    order_price = stock['Open'][n+1]
                    stop_loss_point = order_price + move_stop_loss
            # 多單出場
            elif order_record.GetOpenInterest() == 1:
                if stock['Close'][n] - move_stop_loss > stop_loss_point:
                    stop_loss_point = stock['Close'][n] - move_stop_loss
                elif stock['Close'][n] < stop_loss_point:
                    order_record.Cover('Sell', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)
            # 空單出場
            elif order_record.GetOpenInterest() == -1:
                if stock['Close'][n] + move_stop_loss < stop_loss_point:
                    stop_loss_point = stock['Close'][n] + move_stop_loss
                elif stock['Close'][n] > stop_loss_point:
                    order_record.Cover('Buy', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)

    return order_record

# RSI 指標計算
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

# RSI 交易策略
def rsi_strategy(stock, move_stop_loss):
    stock = calculate_rsi(stock)
    
    order_record = OrderRecord()
    order_price = None
    stop_loss_point = None

    for n in range(1, len(stock)):
        if not np.isnan(stock['RSI'][n-1]):
            # 無未平倉部位
            if order_record.GetOpenInterest() == 0:
                # 多單進場
                if stock['RSI'][n] < 20:
                    order_record.Order('Buy', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)
                    order_price = stock['Open'][n+1]
                    stop_loss_point = order_price - move_stop_loss
                # 空單進場
                elif stock['RSI'][n] > 80:
                    order_record.Order('Sell', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)
                    order_price = stock['Open'][n+1]
                    stop_loss_point = order_price + move_stop_loss
            # 多單出場
            elif order_record.GetOpenInterest() == 1:
                if stock['Close'][n] - move_stop_loss > stop_loss_point:
                    stop_loss_point = stock['Close'][n] - move_stop_loss
                elif stock['Close'][n] < stop_loss_point:
                    order_record.Cover('Sell', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)
            # 空單出場
            elif order_record.GetOpenInterest() == -1:
                if stock['Close'][n] + move_stop_loss < stop_loss_point:
                    stop_loss_point = stock['Close'][n] + move_stop_loss
                elif stock['Close'][n] > stop_loss_point:
                    order_record.Cover('Buy', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)

    return order_record

# MACD 指標計算
def calculate_macd(stock, short_window=12, long_window=26, signal_window=9):
    stock['EMA12'] = stock['Close'].ewm(span=short_window, adjust=False).mean()
    stock['EMA26'] = stock['Close'].ewm(span=long_window, adjust=False).mean()
    stock['MACD'] = stock['EMA12'] - stock['EMA26']
    stock['Signal_Line'] = stock['MACD'].ewm(span=signal_window, adjust=False).mean()
    return stock

# MACD 交易策略
def macd_strategy(stock, move_stop_loss):
    stock = calculate_macd(stock)
    
    order_record = OrderRecord()
    order_price = None
    stop_loss_point = None

    for n in range(1, len(stock)):
        if not np.isnan(stock['MACD'][n-1]):
            # 無未平倉部位
            if order_record.GetOpenInterest() == 0:
                # 多單進場
                if stock['MACD'][n-1] <= stock['Signal_Line'][n-1] and stock['MACD'][n] > stock['Signal_Line'][n]:
                    order_record.Order('Buy', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)
                    order_price = stock['Open'][n+1]
                    stop_loss_point = order_price - move_stop_loss
                # 空單進場
                elif stock['MACD'][n-1] >= stock['Signal_Line'][n-1] and stock['MACD'][n] < stock['Signal_Line'][n]:
                    order_record.Order('Sell', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)
                    order_price = stock['Open'][n+1]
                    stop_loss_point = order_price + move_stop_loss
            # 多單出場
            elif order_record.GetOpenInterest() == 1:
                if stock['Close'][n] - move_stop_loss > stop_loss_point:
                    stop_loss_point = stock['Close'][n] - move_stop_loss
                elif stock['Close'][n] < stop_loss_point:
                    order_record.Cover('Sell', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)
            # 空單出場
            elif order_record.GetOpenInterest() == -1:
                if stock['Close'][n] + move_stop_loss < stop_loss_point:
                    stop_loss_point = stock['Close'][n] + move_stop_loss
                elif stock['Close'][n] > stop_loss_point:
                    order_record.Cover('Buy', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)

    return order_record

# 計算並顯示績效
def evaluate_strategy(order_record):
    st.subheader("交易績效")
    trade_record = order_record.GetTradeRecord()
    profit = order_record.GetProfit()
    total_profit = order_record.GetTotalProfit()
    win_rate = order_record.GetWinRate()
    acc_loss = order_record.GetAccLoss()
    mdd = order_record.GetMDD()

    st.write(f"交易紀錄: {trade_record}")
    st.write(f"損益: {profit}")
    st.write(f"總損益: {total_profit}")
    st.write(f"勝率: {win_rate * 100:.2f}%")
    st.write(f"最大連續虧損: {acc_loss}")
    st.write(f"最大資金回落 (MDD): {mdd}")
