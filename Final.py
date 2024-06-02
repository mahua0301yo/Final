# 載入必要模組
import yfinance as yf
import pandas as pd
import streamlit as st
import streamlit.components.v1 as stc
import plotly.graph_objs as go
import datetime
import numpy as np

# 定義函數來顯示標題
def display_header():
    html_temp = """
        <div style="background-color:#3872fb;padding:10px;border-radius:10px">
        <h1 style="color:white;text-align:center;">股票技術指標分析</h1>
        </div>
    """
    stc.html(html_temp)

# 定義函數來讀取股票數據
def load_stock_data(stockname, start_date, end_date, interval):
    try:
        stock = yf.download(stockname, start=start_date, end=end_date, interval=interval)
        if stock.empty:
            st.error("未能讀取到數據，請檢查股票代號是否正確")
            return None
        else:
            st.success("數據讀取成功")
            stock.rename(columns={'Volume': 'amount'}, inplace=True)
            stock.drop(columns=['Adj Close'], inplace=True)
            stock['Volume'] = (stock['amount'] / (stock['Open'] + stock['Close']) / 2).astype(int)
            cols = stock.columns.tolist()
            vol_idx = cols.index('Volume')
            amt_idx = cols.index('amount')
            cols[vol_idx], cols[amt_idx] = cols[amt_idx], cols[vol_idx]
            stock = stock[cols]
            stock.reset_index(inplace=True)
            return stock
    except Exception as e:
        st.error(f"讀取數據時出錯: {e}")
        return None

# 定義函數來計算KDJ指標
def calculate_kdj(stock, period=14):
    low_min = stock['Low'].rolling(window=period).min()
    high_max = stock['High'].rolling(window=period).max()
    rsv = (stock['Close'] - low_min) / (high_max - low_min) * 100
    stock['K'] = rsv.ewm(com=2).mean()
    stock['D'] = stock['K'].ewm(com=2).mean()
    stock['J'] = 3 * stock['K'] - 2 * stock['D']
    return stock

# 定義函數來計算技術指標，包括布林通道和MACD
def calculate_indicators(stock, bollinger_period, bollinger_std, macd_short_period, macd_long_period, macd_signal_period):
    # 計算布林通道
    stock['Middle_Band'] = stock['Close'].rolling(window=bollinger_period).mean()
    stock['Upper_Band'] = stock['Middle_Band'] + (stock['Close'].rolling(window=bollinger_period).std() * bollinger_std)
    stock['Lower_Band'] = stock['Middle_Band'] - (stock['Close'].rolling(window=bollinger_period).std() * bollinger_std)

    # 計算MACD
    stock['EMA_short'] = stock['Close'].ewm(span=macd_short_period, adjust=False).mean()
    stock['EMA_long'] = stock['Close'].ewm(span=macd_long_period, adjust=False).mean()
    stock['MACD'] = stock['EMA_short'] - stock['EMA_long']
    stock['Signal_Line'] = stock['MACD'].ewm(span=macd_signal_period, adjust=False).mean()

    # 計算KDJ指標
    stock = calculate_kdj(stock)

    return stock

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

# 建立交易策略
def trading_strategy(stock, long_ma_period, short_ma_period, move_stop_loss):
    stock['MA_long'] = stock['Close'].rolling(window=long_ma_period).mean()
    stock['MA_short'] = stock['Close'].rolling(window=short_ma_period).mean()

    order_record = OrderRecord()
    order_price = None
    stop_loss_point = None

    for n in range(1, len(stock)):
        if not np.isnan(stock['MA_long'][n-1]):
            # 無未平倉部位
            if order_record.GetOpenInterest() == 0:
                # 多單進場
                if stock['MA_short'][n-1] <= stock['MA_long'][n-1] and stock['MA_short'][n] > stock['MA_long'][n]:
                    order_record.Order('Buy', stock['Date'][n+1], stock['Date'][n+1], stock['Open'][n+1], 1)
                    order_price = stock['Open'][n+1]
                    stop_loss_point = order_price - move_stop_loss
                # 空單進場
                elif stock['MA_short'][n-1] >= stock['MA_long'][n-1] and stock['MA_short'][n] < stock['MA_long'][n]:
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

# 繪製股票數據和技術指標
def plot_stock_data(stock, order_record):
    # 繪製K線圖與布林通道
    fig_kline = go.Figure()
    fig_kline.add_trace(go.Candlestick(x=stock['Date'], open=stock['Open'], high=stock['High'], low=stock['Low'], close=stock['Close'], name='價格'))
    fig_kline.add_trace(go.Scatter(x=stock['Date'], y=stock['Middle_Band'], line=dict(color='blue', width=1), name='中軌'))
    fig_kline.add_trace(go.Scatter(x=stock['Date'], y=stock['Upper_Band'], line=dict(color='red', width=1), name='上軌'))
    fig_kline.add_trace(go.Scatter(x=stock['Date'], y=stock['Lower_Band'], line=dict(color='green', width=1), name='下軌'))

    # 繪製買賣點
    for trade in order_record.GetTradeRecord():
        color = 'green' if trade['action'] == 'Buy' else 'red'
        symbol = 'arrow-up' if trade['action'] == 'Buy' else 'arrow-down'
        fig_kline.add_trace(go.Scatter(
            x=[trade['time']],
            y=[trade['price']],
            mode='markers',
            marker=dict(color=color, size=15, symbol=symbol),
            name=trade['action']
        ))

    fig_kline.update_layout(title='K線圖與布林通道', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_kline, use_container_width=True)

    # 繪製MACD
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=stock['Date'], y=stock['MACD'], line=dict(color='blue', width=1), name='MACD'))
    fig_macd.add_trace(go.Scatter(x=stock['Date'], y=stock['Signal_Line'], line=dict(color='orange', width=1), name='信號線'))
    fig_macd.add_trace(go.Bar(x=stock['Date'], y=stock['MACD'] - stock['Signal_Line'], name='柱狀圖'))

    fig_macd.update_layout(title='MACD', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_macd, use_container_width=True)

    # 繪製KDJ
    fig_kdj = go.Figure()
    fig_kdj.add_trace(go.Scatter(x=stock['Date'], y=stock['K'], line=dict(color='blue', width=1), name='K'))
    fig_kdj.add_trace(go.Scatter(x=stock['Date'], y=stock['D'], line=dict(color='orange', width=1), name='D'))
    fig_kdj.add_trace(go.Scatter(x=stock['Date'], y=stock['J'], line=dict(color='green', width=1), name='J'))

    fig_kdj.update_layout(title='KDJ', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_kdj, use_container_width=True)

    # 計算並顯示績效
    st.subheader("交易績效")
    trade_record = order_record.GetTradeRecord()
    profit = order_record.GetProfit()
    total_profit = order_record.GetTotalProfit()
    win_rate = order_record.GetWinRate()
    acc_loss = order_record.GetAccLoss()
    mdd = order_record.GetMDD()

    st.write(f"損益: {profit}")
    st.write(f"總損益: {total_profit}")
    st.write(f"勝率: {win_rate * 100:.2f}%")
    st.write(f"最大連續虧損: {acc_loss}")
    st.write(f"最大資金回落 (MDD): {mdd}")

# 主函數
def main():
    display_header()

    # 選擇資料區間
    st.subheader("選擇資料區間")
    start_date = st.date_input('選擇開始日期', datetime.date(2015, 1, 1))
    end_date = st.date_input('選擇結束日期', datetime.date(2100, 12, 31))
    stockname = st.text_input('請輸入股票代號 (例: 2330.TW)', '2330.TW')

    # 選擇K線時間長
    interval_options = {
        "1天": "1d",
        "1星期": "1wk",
        "1個月": "1mo",
        "3個月": "3mo"
    }
    interval_label = st.selectbox("選擇K線時間長", list(interval_options.keys()))
    interval = interval_options[interval_label]

    # 輸入布林通道的週期和標準差倍數
    bollinger_period = st.number_input('請輸入布林通道週期', min_value=1, max_value=100, value=20, step=1)
    bollinger_std = st.number_input('請輸入布林通道標準差倍數', min_value=0.1, max_value=10.0, value=2.0, step=0.1)

    # 輸入MACD的參數
    macd_short_period = st.number_input('請輸入MACD短期EMA週期', min_value=1, max_value=50, value=12, step=1)
    macd_long_period = st.number_input('請輸入MACD長期EMA週期', min_value=1, max_value=50, value=26, step=1)
    macd_signal_period = st.number_input('請輸入MACD信號線週期', min_value=1, max_value=50, value=9, step=1)

    # 輸入移動停損點數和均線參數
    long_ma_period = st.number_input('請輸入長期均線週期', min_value=1, max_value=100, value=20, step=1)
    short_ma_period = st.number_input('請輸入短期均線週期', min_value=1, max_value=100, value=10, step=1)
    move_stop_loss = st.number_input('請輸入移動停損點數', min_value=1, max_value=100, value=30, step=1)

    # 驗證日期輸入
    if start_date > end_date:
        st.error("開始日期不能晚於結束日期")
    else:
        stock = load_stock_data(stockname, start_date, end_date, interval)
        if stock is not None:
            stock = calculate_indicators(stock, bollinger_period, bollinger_std, macd_short_period, macd_long_period, macd_signal_period)
            order_record = trading_strategy(stock, long_ma_period, short_ma_period, move_stop_loss)
            plot_stock_data(stock, order_record)

if __name__ == "__main__":
    main()
