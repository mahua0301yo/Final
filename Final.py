import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np

# 交易記錄類別
class OrderRecord:
    def __init__(self):
        self.trades = []
        self.open_interest = 0
        self.profits = []

    def Order(self, action, product, time, price, quantity):
        self.trades.append({
            'action': action,
            'product': product,
            'time': time,
            'price': price,
            'quantity': quantity
        })
        self.open_interest += quantity if action == 'Buy' else -quantity

    def Cover(self, action, product, time, price, quantity):
        self.trades.append({
            'action': action,
            'product': product,
            'time': time,
            'price': price,
            'quantity': -quantity
        })
        self.open_interest -= quantity if action == 'Sell' else -quantity
        # 計算損益
        last_trade = self.trades[-2]
        if last_trade['action'] == 'Buy':
            self.profits.append(price - last_trade['price'])
        elif last_trade['action'] == 'Sell':
            self.profits.append(last_trade['price'] - price)

    def GetOpenInterest(self):
        return self.open_interest

    def GetTradeRecord(self):
        return self.trades

    def GetProfit(self):
        return self.profits

    def GetTotalProfit(self):
        return sum(self.profits)

    def GetWinRate(self):
        wins = [p for p in self.profits if p > 0]
        return len(wins) / len(self.profits) if self.profits else 0

    def GetAccLoss(self):
        acc_loss = 0
        max_acc_loss = 0
        for p in self.profits:
            if p < 0:
                acc_loss += p
                if acc_loss < max_acc_loss:
                    max_acc_loss = acc_loss
            else:
                acc_loss = 0
        return max_acc_loss

    def GetMDD(self):
        equity_curve = np.cumsum(self.profits)
        drawdowns = equity_curve - np.maximum.accumulate(equity_curve)
        return drawdowns.min() if drawdowns.size > 0 else 0

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

# 定義函數來計算布林通道指標及交易策略
def calculate_bollinger_bands_strategy(stock, period=20, std_dev=2):
    order_record = OrderRecord()

    stock['Middle_Band'] = stock['Close'].rolling(window=period).mean()
    stock['Upper_Band'] = stock['Middle_Band'] + std_dev * stock['Close'].rolling(window=period).std()
    stock['Lower_Band'] = stock['Middle_Band'] - std_dev * stock['Close'].rolling(window=period).std()

    # 簡單的交易策略示例：當股價突破上軌道做多，突破下軌道做空
    for i in range(1, len(stock)):
        if stock['Close'][i] > stock['Upper_Band'][i-1] and stock['Close'][i-1] <= stock['Upper_Band'][i-1]:
            order_record.Order('Buy', 'Stock', stock['Date'][i], stock['Close'][i], 100)  # 假設固定買入100股
        elif stock['Close'][i] < stock['Lower_Band'][i-1] and stock['Close'][i-1] >= stock['Lower_Band'][i-1]:
            order_record.Cover('Sell', 'Stock', stock['Date'][i], stock['Close'][i], 100)  # 假設固定賣出100股

    # 計算策略績效指標
    total_profit = order_record.GetTotalProfit()
    win_rate = order_record.GetWinRate()
    max_drawdown = order_record.GetMDD()

    print(f"布林通道策略總損益: {total_profit}")
    print(f"布林通道策略勝率: {win_rate}")
    print(f"布林通道策略最大回撤: {max_drawdown}")

    return stock, order_record

# 定義函數來計算KDJ指標及交易策略
def calculate_kdj_strategy(stock, period=14):
    order_record = OrderRecord()

    low_min = stock['Low'].rolling(window=period).min()
    high_max = stock['High'].rolling(window=period).max()
    rsv = (stock['Close'] - low_min) / (high_max - low_min) * 100
    stock['K'] = rsv.ewm(com=2).mean()
    stock['D'] = stock['K'].ewm(com=2).mean()
    stock['J'] = 3 * stock['K'] - 2 * stock['D']

    # 簡單的交易策略示例：當K超過80做空，低於20做多
    for i in range(1, len(stock)):
        if stock['K'][i] > 80 and stock['K'][i-1] <= 80:
            order_record.Cover('Sell', 'Stock', stock['Date'][i], stock['Close'][i], 100)  # 假設固定賣出100股
        elif stock['K'][i] < 20 and stock['K'][i-1] >= 20:
            order_record.Order('Buy', 'Stock', stock['Date'][i], stock['Close'][i], 100)  # 假設固定買入100股

    # 計算策略績效指標
    total_profit = order_record.GetTotalProfit()
    win_rate = order_record.GetWinRate()
    max_drawdown = order_record.GetMDD()

    print(f"KDJ策略總損益: {total_profit}")
    print(f"KDJ策略勝率: {win_rate}")
    print(f"KDJ策略最大回撤: {max_drawdown}")

    return stock, order_record

# 定義函數來計算RSI指標及交易策略
def calculate_rsi_strategy(stock, period=14):
    order_record = OrderRecord()

    delta = stock['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    stock['RSI'] = 100 - (100 / (1 + rs))

    # 簡單的交易策略示例：當RSI超過70做空，低於30做多
    for i in range(1, len(stock)):
        if stock['RSI'][i] > 70 and stock['RSI'][i-1] <= 70:
            order_record.Cover('Sell', 'Stock', stock['Date'][i], stock['Close'][i], 100)  # 假設固定賣出100股
        elif stock['RSI'][i] < 30 and stock['RSI'][i-1] >= 30:
            order_record.Order('Buy', 'Stock', stock['Date'][i], stock['Close'][i], 100)  # 假設固定買入100股

    # 計算策略績效指標
    total_profit = order_record.GetTotalProfit()
    win_rate = order_record.GetWinRate()
    max_drawdown = order_record.GetMDD()

    print(f"RSI策略總損益: {total_profit}")
    print(f"RSI策略勝率: {win_rate}")
    print(f"RSI策略最大回撤: {max_drawdown}")

    return stock, order_record

# 定義函數來計算MACD指標及交易策略
def calculate_macd_strategy(stock, slow=26, fast=12, signal=9):
    order_record = OrderRecord()

    exp1 = stock['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = stock['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    stock['MACD'] = macd - signal_line

    # 簡單的交易策略示例：當MACD超過信號線做多，低於信號線做空
    for i in range(1, len(stock)):
        if stock['MACD'][i] > 0 and stock['MACD'][i-1] <= 0:
            order_record.Order('Buy', 'Stock', stock['Date'][i], stock['Close'][i], 100)  # 假設固定買入100股
        elif stock['MACD'][i] < 0 and stock['MACD'][i-1] >= 0:
            order_record.Cover('Sell', 'Stock', stock['Date'][i], stock['Close'][i], 100)  # 假設固定賣出100股

    # 計算策略績效指標
    total_profit = order_record.GetTotalProfit()
    win_rate = order_record.GetWinRate()
    max_drawdown = order_record.GetMDD()

    print(f"MACD策略總損益: {total_profit}")
    print(f"MACD策略勝率: {win_rate}")
    print(f"MACD策略最大回撤: {max_drawdown}")

    return stock, order_record

# 定義函數來計算唐奇安通道指標及交易策略
def calculate_donchian_channel_strategy(stock, period=20):
    order_record = OrderRecord()

    stock['Upper_Channel'] = stock['High'].rolling(window=period).max()
    stock['Lower_Channel'] = stock['Low'].rolling(window=period).min()

    # 簡單的交易策略示例：當股價突破上通道做多，突破下通道做空
    for i in range(1, len(stock)):
        if stock['Close'][i] > stock['Upper_Channel'][i-1] and stock['Close'][i-1] <= stock['Upper_Channel'][i-1]:
            order_record.Order('Buy', 'Stock', stock['Date'][i], stock['Close'][i], 100)  # 假設固定買入100股
        elif stock['Close'][i] < stock['Lower_Channel'][i-1] and stock['Close'][i-1] >= stock['Lower_Channel'][i-1]:
            order_record.Cover('Sell', 'Stock', stock['Date'][i], stock['Close'][i], 100)  # 假設固定賣出100股

    # 計算策略績效指標
    total_profit = order_record.GetTotalProfit()
    win_rate = order_record.GetWinRate()
    max_drawdown = order_record.GetMDD()

    print(f"唐奇安通道策略總損益: {total_profit}")
    print(f"唐奇安通道策略勝率: {win_rate}")
    print(f"唐奇安通道策略最大回撤: {max_drawdown}")

    return stock, order_record

# 主程式流程
def main():
    st.title('技術指標交易策略回測')

    # 輸入股票代號和日期範圍
    stockname = st.sidebar.text_input("請輸入股票代號（例如：AAPL）", value='AAPL')
    start_date = st.sidebar.date_input("請選擇起始日期")
    end_date = st.sidebar.date_input("請選擇結束日期")
    interval = st.sidebar.selectbox("請選擇數據頻率", ('1d', '1wk', '1mo'), index=0)

    # 選擇策略
    strategy_name = st.sidebar.selectbox("請選擇交易策略", ('布林通道', 'KDJ', 'RSI', 'MACD', '唐奇安通道'))

    # 讀取股票數據
    stock = load_stock_data(stockname, start_date, end_date, interval)
    if stock is not None:
        if strategy_name == "布林通道":
            bollinger_period = st.sidebar.slider("布林通道週期", min_value=5, max_value=50, value=20, step=1)
            bollinger_std = st.sidebar.slider("布林通道標準差倍數", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
            stock, order_record = calculate_bollinger_bands_strategy(stock, period=bollinger_period, std_dev=bollinger_std)

        elif strategy_name == "KDJ":
            kdj_period = st.sidebar.slider("KDJ週期", min_value=5, max_value=50, value=14, step=1)
            stock, order_record = calculate_kdj_strategy(stock, period=kdj_period)

        elif strategy_name == "RSI":
            rsi_period = st.sidebar.slider("RSI週期", min_value=5, max_value=50, value=14, step=1)
            stock, order_record = calculate_rsi_strategy(stock, period=rsi_period)

        elif strategy_name == "MACD":
            macd_slow = st.sidebar.slider("MACD慢線週期", min_value=5, max_value=50, value=26, step=1)
            macd_fast = st.sidebar.slider("MACD快線週期", min_value=1, max_value=25, value=12, step=1)
            macd_signal = st.sidebar.slider("MACD信號線週期", min_value=1, max_value=20, value=9, step=1)
            stock, order_record = calculate_macd_strategy(stock, slow=macd_slow, fast=macd_fast, signal=macd_signal)

        elif strategy_name == "唐奇安通道":
            donchian_period = st.sidebar.slider("唐奇安通道週期", min_value=5, max_value=50, value=20, step=1)
            stock, order_record = calculate_donchian_channel_strategy(stock, period=donchian_period)

        # 繪製股價走勢和指標
        plot_stock_data(stock, strategy_name)

        # 顯示交易記錄和績效指標
        st.subheader("交易記錄")
        st.write(order_record.GetTradeRecord())

        st.subheader("績效指標")
        st.write(f"總損益: {order_record.GetTotalProfit()}")
        st.write(f"勝率: {order_record.GetWinRate()}")
        st.write(f"最大回撤: {order_record.GetMDD()}")

# 定義函數來繪製股價走勢和指標
def plot_stock_data(stock, strategy_name):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=(f'{strategy_name} 策略股價走勢', f'{strategy_name} 相關指標'))

    fig.add_trace(go.Candlestick(x=stock['Date'],
                                 open=stock['Open'], high=stock['High'],
                                 low=stock['Low'], close=stock['Close'],
                                 name='股價走勢'), row=1, col=1)

    if strategy_name == '布林通道':
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Upper_Band'], mode='lines', name='布林通道上軌'),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Middle_Band'], mode='lines', name='布林通道中軌'),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Lower_Band'], mode='lines', name='布林通道下軌'),
                      row=1, col=1)

    elif strategy_name == 'KDJ':
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['K'], mode='lines', name='K'),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['D'], mode='lines', name='D'),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['J'], mode='lines', name='J'),
                      row=2, col=1)

    elif strategy_name == 'RSI':
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['RSI'], mode='lines', name='RSI'),
                      row=2, col=1)

    elif strategy_name == 'MACD':
        fig.add_trace(go.Bar(x=stock['Date'], y=stock['MACD'], name='MACD'),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Close'], mode='lines', name='Close'),
                      row=2, col=1)

    elif strategy_name == '唐奇安通道':
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Upper_Channel'], mode='lines', name='唐奇安通道上通道'),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Lower_Channel'], mode='lines', name='唐奇安通道下通道'),
                      row=1, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False, height=800)
    st.plotly_chart(fig)

if __name__ == '__main__':
    main()
