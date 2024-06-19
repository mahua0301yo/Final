import yfinance as yf
import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Define OrderRecord class
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

# Function to load stock data
def load_stock_data(stockname, start_date, end_date, interval):
    stock = yf.download(stockname, start=start_date, end=end_date, interval=interval)
    if stock.empty:
        st.error("Unable to fetch data. Please check if the stock symbol is correct.")
        return None
    stock.rename(columns={'Volume': 'amount'}, inplace=True)
    stock.drop(columns=['Adj Close'], inplace=True)
    stock['Volume'] = (stock['amount'] / (stock['Open'] + stock['Close']) / 2).astype(int)
    stock.reset_index(inplace=True)
    return stock

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(stock, period=20, std_dev=2):
    stock['Middle_Band'] = stock['Close'].rolling(window=period).mean()
    stock['Upper_Band'] = stock['Middle_Band'] + std_dev * stock['Close'].rolling(window=period).std()
    stock['Lower_Band'] = stock['Middle_Band'] - std_dev * stock['Close'].rolling(window=period).std()
    return stock

# Bollinger Bands trading strategy
def bollinger_bands_strategy(stock, move_stop_loss):
    stock = calculate_bollinger_bands(stock)
    
    order_record = OrderRecord()
    order_price = None
    stop_loss_point = None

    for n in range(1, len(stock)):
        if not np.isnan(stock['Middle_Band'][n-1]):
            # No open positions
            if order_record.GetOpenInterest() == 0:
                # Enter long position
                if stock['Close'][n] < stock['Lower_Band'][n]:
                    order_record.Order('Buy', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)
                    order_price = stock['Open'][n+1]
                    stop_loss_point = order_price - move_stop_loss
                # Enter short position
                elif stock['Close'][n] > stock['Upper_Band'][n]:
                    order_record.Order('Sell', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)
                    order_price = stock['Open'][n+1]
                    stop_loss_point = order_price + move_stop_loss
            # Exit long position
            elif order_record.GetOpenInterest() == 1:
                if stock['Close'][n] - move_stop_loss > stop_loss_point:
                    stop_loss_point = stock['Close'][n] - move_stop_loss
                elif stock['Close'][n] < stop_loss_point:
                    order_record.Cover('Sell', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)
            # Exit short position
            elif order_record.GetOpenInterest() == -1:
                if stock['Close'][n] + move_stop_loss < stop_loss_point:
                    stop_loss_point = stock['Close'][n] + move_stop_loss
                elif stock['Close'][n] > stop_loss_point:
                    order_record.Cover('Buy', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)

    return order_record

# Function to calculate KDJ indicator
def calculate_kdj(stock, period=14):
    low_min = stock['Low'].rolling(window=period).min()
    high_max = stock['High'].rolling(window=period).max()
    rsv = (stock['Close'] - low_min) / (high_max - low_min) * 100
    stock['K'] = rsv.ewm(com=2).mean()
    stock['D'] = stock['K'].ewm(com=2).mean()
    stock['J'] = 3 * stock['K'] - 2 * stock['D']
    return stock

# KDJ trading strategy
def kdj_strategy(stock, move_stop_loss):
    stock = calculate_kdj(stock)
    
    order_record = OrderRecord()
    order_price = None
    stop_loss_point = None

    for n in range(1, len(stock)):
        if not np.isnan(stock['K'][n-1]):
            # No open positions
            if order_record.GetOpenInterest() == 0:
                # Enter long position
                if stock['J'][n-1] < stock['K'][n-1] and stock['J'][n] > stock['K'][n]:
                    order_record.Order('Buy', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)
                    order_price = stock['Open'][n+1]
                    stop_loss_point = order_price - move_stop_loss
                # Enter short position
                elif stock['J'][n-1] > stock['K'][n-1] and stock['J'][n] < stock['K'][n]:
                    order_record.Order('Sell', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)
                    order_price = stock['Open'][n+1]
                    stop_loss_point = order_price + move_stop_loss
            # Exit long position
            elif order_record.GetOpenInterest() == 1:
                if stock['Close'][n] - move_stop_loss > stop_loss_point:
                    stop_loss_point = stock['Close'][n] - move_stop_loss
                elif stock['Close'][n] < stop_loss_point:
                    order_record.Cover('Sell', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)
            # Exit short position
            elif order_record.GetOpenInterest() == -1:
                if stock['Close'][n] + move_stop_loss < stop_loss_point:
                    stop_loss_point = stock['Close'][n] + move_stop_loss
                elif stock['Close'][n] > stop_loss_point:
                    order_record.Cover('Buy', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)

    return order_record

# Function to calculate RSI indicator
def calculate_rsi(stock, period=14):
    delta = stock['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    stock['RSI'] = 100 - (100 / (1 + rs))

    # Add overbought and oversold markers
    stock['Overbought'] = stock['RSI'] > 80
    stock['Oversold'] = stock['RSI'] < 20

    return stock

# RSI trading strategy
def rsi_strategy(stock, move_stop_loss):
    stock = calculate_rsi(stock)
    
    order_record = OrderRecord()
    order_price = None
    stop_loss_point = None

    for n in range(1, len(stock)):
        if not np.isnan(stock['RSI'][n-1]):
            # No open positions
            if order_record.GetOpenInterest() == 0:
                # Enter long position
                if stock['RSI'][n-1] < 30 and stock['RSI'][n] > 30:
                    order_record.Order('Buy', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)
                    order_price = stock['Open'][n+1]
                    stop_loss_point = order_price - move_stop_loss
                # Enter short position
                elif stock['RSI'][n-1] > 70 and stock['RSI'][n] < 70:
                    order_record.Order('Sell', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)
                    order_price = stock['Open'][n+1]
                    stop_loss_point = order_price + move_stop_loss
            # Exit long position
            elif order_record.GetOpenInterest() == 1:
                if stock['Close'][n] - move_stop_loss > stop_loss_point:
                    stop_loss_point = stock['Close'][n] - move_stop_loss
                elif stock['Close'][n] < stop_loss_point:
                    order_record.Cover('Sell', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)
            # Exit short position
            elif order_record.GetOpenInterest() == -1:
                if stock['Close'][n] + move_stop_loss < stop_loss_point:
                    stop_loss_point = stock['Close'][n] + move_stop_loss
                elif stock['Close'][n] > stop_loss_point:
                    order_record.Cover('Buy', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)

    return order_record

# Function to calculate MACD indicator
def calculate_macd(stock, short_period=12, long_period=26, signal_period=9):
    stock['ShortEMA'] = stock['Close'].ewm(span=short_period, min_periods=1).mean()
    stock['LongEMA'] = stock['Close'].ewm(span=long_period, min_periods=1).mean()
    stock['MACD'] = stock['ShortEMA'] - stock['LongEMA']
    stock['Signal_Line'] = stock['MACD'].ewm(span=signal_period, min_periods=1).mean()
    return stock

# MACD trading strategy
def macd_strategy(stock, move_stop_loss):
    stock = calculate_macd(stock)
    
    order_record = OrderRecord()
    order_price = None
    stop_loss_point = None

    for n in range(1, len(stock)):
        if not np.isnan(stock['MACD'][n-1]):
            # No open positions
            if order_record.GetOpenInterest() == 0:
                # Enter long position
                if stock['MACD'][n-1] < stock['Signal_Line'][n-1] and stock['MACD'][n] > stock['Signal_Line'][n]:
                    order_record.Order('Buy', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)
                    order_price = stock['Open'][n+1]
                    stop_loss_point = order_price - move_stop_loss
                # Enter short position
                elif stock['MACD'][n-1] > stock['Signal_Line'][n-1] and stock['MACD'][n] < stock['Signal_Line'][n]:
                    order_record.Order('Sell', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)
                    order_price = stock['Open'][n+1]
                    stop_loss_point = order_price + move_stop_loss
            # Exit long position
            elif order_record.GetOpenInterest() == 1:
                if stock['Close'][n] - move_stop_loss > stop_loss_point:
                    stop_loss_point = stock['Close'][n] - move_stop_loss
                elif stock['Close'][n] < stop_loss_point:
                    order_record.Cover('Sell', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)
            # Exit short position
            elif order_record.GetOpenInterest() == -1:
                if stock['Close'][n] + move_stop_loss < stop_loss_point:
                    stop_loss_point = stock['Close'][n] + move_stop_loss
                elif stock['Close'][n] > stop_loss_point:
                    order_record.Cover('Buy', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)

    return order_record

# Main function to run the Streamlit app
def main():
    st.title('股票交易策略回測')

    st.sidebar.header('選擇股票和日期範圍')
    stockname = st.sidebar.text_input('輸入股票代碼 (例如 AAPL, MSFT)', value='AAPL')
    start_date = st.sidebar.date_input('開始日期', value=pd.to_datetime('2021-01-01'))
    end_date = st.sidebar.date_input('結束日期', value=pd.to_datetime('2022-01-01'))
    interval = st.sidebar.selectbox('交易間隔', ['1d', '1wk', '1mo'], index=0)

    strategies = {
        '布林通道': bollinger_bands_strategy,
        'KDJ': kdj_strategy,
        'RSI': rsi_strategy,
        'MACD': macd_strategy,
    }

    selected_strategy = st.sidebar.selectbox('選擇交易策略', list(strategies.keys()))

    move_stop_loss = st.sidebar.number_input('移動止損點數', value=0.5, format='%.1f')

    stock = load_stock_data(stockname, start_date, end_date, interval)

    if stock is not None:
        order_record = strategies[selected_strategy](stock.copy(), move_stop_loss)

        st.subheader(f"{stockname} {selected_strategy} 策略交易結果")

        st.write(f"總利潤: {order_record.GetTotalProfit():.2f}")
        st.write(f"勝率: {order_record.GetWinRate() * 100:.2f}%")
        st.write(f"最大連續虧損: {order_record.GetMDD():.2f}")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)

        fig.add_trace(go.Candlestick(x=stock['Date'], open=stock['Open'], high=stock['High'], low=stock['Low'], close=stock['Close'], name='K線圖'), row=1, col=1)

        if selected_strategy == '布林通道':
            fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Upper_Band'], line=dict(color='blue', width=1), name='上軌線'), row=1, col=1)
            fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Middle_Band'], line=dict(color='black', width=1), name='中軌線'), row=1, col=1)
            fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Lower_Band'], line=dict(color='red', width=1), name='下軌線'), row=1, col=1)
        elif selected_strategy == 'KDJ':
            fig.add_trace(go.Scatter(x=stock['Date'], y=stock['K'], line=dict(color='blue', width=1), name='K線'), row=2, col=1)
            fig.add_trace(go.Scatter(x=stock['Date'], y=stock['D'], line=dict(color='green', width=1), name='D線'), row=2, col=1)
            fig.add_trace(go.Scatter(x=stock['Date'], y=stock['J'], line=dict(color='red', width=1), name='J線'), row=2, col=1)
        elif selected_strategy == 'RSI':
            fig.add_trace(go.Scatter(x=stock['Date'], y=stock['RSI'], line=dict(color='blue', width=1), name='RSI'), row=2, col=1)
            fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Overbought'] * 100, line=dict(color='red', width=1), name='超買'), row=2, col=1)
            fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Oversold'] * 100, line=dict(color='green', width=1), name='超賣'), row=2, col=1)
        elif selected_strategy == 'MACD':
            fig.add_trace(go.Scatter(x=stock['Date'], y=stock['MACD'], line=dict(color='blue', width=1), name='MACD'), row=2, col=1)
            fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Signal_Line'], line=dict(color='red', width=1), name='信號線'), row=2, col=1)

        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig)

        st.subheader("交易記錄")
        trade_record = order_record.GetTradeRecord()
        if trade_record:
            trade_df = pd.DataFrame(trade_record)
            st.dataframe(trade_df)

# Run the main function
if __name__ == "__main__":
    main()
