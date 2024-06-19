import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# 讀取股票數據
def load_stock_data(stockname, start_date, end_date, interval):
    data = yf.download(stockname, start=start_date, end=end_date, interval=interval)
    data.reset_index(inplace=True)
    return data

# 定義函數來計算布林通道指標及交易策略
def calculate_bollinger_bands_strategy(stock, period=20, std_dev=2.0):
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

    low_list = stock['Low'].rolling(window=period).min()
    high_list = stock['High'].rolling(window=period).max()

    rsv = (stock['Close'] - low_list) / (high_list - low_list) * 100
    stock['K'] = rsv.ewm(com=2).mean()
    stock['D'] = stock['K'].ewm(com=2).mean()
    stock['J'] = 3 * stock['K'] - 2 * stock['D']

    # 簡單的交易策略示例：當KDJ的K穿越D做多，K穿越D做空
    for i in range(1, len(stock)):
        if stock['K'][i] > stock['D'][i] and stock['K'][i-1] <= stock['D'][i-1]:
            order_record.Order('Buy', 'Stock', stock['Date'][i], stock['Close'][i], 100)  # 假設固定買入100股
        elif stock['K'][i] < stock['D'][i] and stock['K'][i-1] >= stock['D'][i-1]:
            order_record.Cover('Sell', 'Stock', stock['Date'][i], stock['Close'][i], 100)  # 假設固定賣出100股

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

    RS = gain / loss
    stock['RSI'] = 100 - (100 / (1 + RS))

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
def calculate_macd_strategy(stock, short_window=12, long_window=26, signal_window=9):
    order_record = OrderRecord()

    exp1 = stock['Close'].ewm(span=short_window, adjust=False).mean()
    exp2 = stock['Close'].ewm(span=long_window, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()

    stock['MACD'] = macd_line - signal_line

    # 簡單的交易策略示例：MACD的金叉做多，死叉做空
    for i in range(1, len(stock)):
        if macd_line[i] > signal_line[i] and macd_line[i-1] <= signal_line[i-1]:
            order_record.Order('Buy', 'Stock', stock['Date'][i], stock['Close'][i], 100)  # 假設固定買入100股
        elif macd_line[i] < signal_line[i] and macd_line[i-1] >= signal_line[i-1]:
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

# 繪製股票走勢和技術指標
def plot_stock_data(stock, strategy_name):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=('股價走勢', strategy_name))

    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Close'], mode='lines', name='股價走勢'), row=1, col=1)

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

# 主程式流程
def main():
    st.title('股票技術指標分析與交易策略回測')

    # 輸入股票代號
    stockname = st.sidebar.text_input('請輸入股票代號（如AAPL）：', 'AAPL')

    # 輸入日期範圍
    start_date = st.sidebar.text_input('請輸入起始日期（YYYY-MM-DD）：', '2020-01-01')
    end_date = st.sidebar.text_input('請輸入結束日期（YYYY-MM-DD）：', '2023-01-01')

    # 選擇交易策略
    strategy_name = st.sidebar.selectbox('選擇交易策略：', ['布林通道', 'KDJ', 'RSI', 'MACD', '唐奇安通道'])

    # 讀取股票數據
    try:
        stock = load_stock_data(stockname, start_date, end_date, '1d')
    except Exception as e:
        st.error(f'載入資料時發生錯誤：{e}')
        return

    st.subheader(f'{stockname} 股票 {start_date} 到 {end_date} 的{strategy_name}策略回測')

    if strategy_name == '布林通道':
        stock, order_record = calculate_bollinger_bands_strategy(stock)
    elif strategy_name == 'KDJ':
        stock, order_record = calculate_kdj_strategy(stock)
    elif strategy_name == 'RSI':
        stock, order_record = calculate_rsi_strategy(stock)
    elif strategy_name == 'MACD':
        stock, order_record = calculate_macd_strategy(stock)
    elif strategy_name == '唐奇安通道':
        stock, order_record = calculate_donchian_channel_strategy(stock)

    # 顯示交易記錄
    trades = order_record.GetTradeRecord()
    st.subheader('交易記錄')
    st.write(pd.DataFrame(trades))

    # 顯示績效指標
    total_profit = order_record.GetTotalProfit()
    win_rate = order_record.GetWinRate()
    max_drawdown = order_record.GetMDD()

    st.subheader('績效指標')
    st.write(f"總損益: {total_profit}")
    st.write(f"勝率: {win_rate}")
    st.write(f"最大回撤: {max_drawdown}")

    # 繪製股票走勢和技術指標
    plot_stock_data(stock, strategy_name)

if __name__ == '__main__':
    main()
