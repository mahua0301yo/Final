# 載入必要模組
import yfinance as yf
import pandas as pd
import streamlit as st
import streamlit.components.v1 as stc
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import datetime

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

# 繪製股票數據和技術指標
def plot_stock_data(stock):
    # 繪製K線圖與布林通道
    fig_kline = go.Figure()
    fig_kline.add_trace(go.Candlestick(x=stock['Date'], open=stock['Open'], high=stock['High'], low=stock['Low'], close=stock['Close'], name='價格'))
    fig_kline.add_trace(go.Scatter(x=stock['Date'], y=stock['Middle_Band'], line=dict(color='blue', width=1), name='中軌'))
    fig_kline.add_trace(go.Scatter(x=stock['Date'], y=stock['Upper_Band'], line=dict(color='red', width=1), name='上軌'))
    fig_kline.add_trace(go.Scatter(x=stock['Date'], y=stock['Lower_Band'], line=dict(color='green', width=1), name='下軌'))

    fig_kline.update_layout(title='股票價格與布林通道',
                            xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_kline, use_container_width=True)

    # 繪製MACD
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=stock['Date'], y=stock['MACD'], line=dict(color='blue', width=1), name='MACD'))
    fig_macd.add_trace(go.Scatter(x=stock['Date'], y=stock['Signal_Line'], line=dict(color='red', width=1), name='信號線'))
    fig_macd.add_trace(go.Bar(x=stock['Date'], y=stock['MACD'] - stock['Signal_Line'], name='柱狀圖'))

    fig_macd.update_layout(title='MACD',
                           xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_macd, use_container_width=True)

    # 繪製KDJ
    fig_kdj = go.Figure()
    fig_kdj.add_trace(go.Scatter(x=stock['Date'], y=stock['K'], line=dict(color='blue', width=1), name='K'))
    fig_kdj.add_trace(go.Scatter(x=stock['Date'], y=stock['D'], line=dict(color='orange', width=1), name='D'))
    fig_kdj.add_trace(go.Scatter(x=stock['Date'], y=stock['J'], line=dict(color='green', width=1), name='J'))

    fig_kdj.update_layout(title='KDJ',
                          xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_kdj, use_container_width=True)
class OrderRecord:
    def __init__(self):
        self.records = []
        self.open_interest = 0
        self.total_profit = 0
        self.trades = []

    def Order(self, action, product, time, price, qty):
        self.records.append((action, product, time, price, qty))
        if action == 'Buy':
            self.open_interest += qty
        elif action == 'Sell':
            self.open_interest -= qty

    def Cover(self, action, product, time, price, qty):
        self.records.append((action, product, time, price, qty))
        if action == 'Buy':
            self.open_interest += qty
        elif action == 'Sell':
            self.open_interest -= qty
        self.trades.append((product, time, price, qty))
        self.total_profit += price - self.records[-1][3]

    def GetOpenInterest(self):
        return self.open_interest

    def GetTradeRecord(self):
        return self.trades

    def GetProfit(self):
        return self.total_profit

    def GetTotalProfit(self):
        return self.total_profit

    def GetWinRate(self):
        wins = [t for t in self.trades if t[3] > 0]
        return len(wins) / len(self.trades) if self.trades else 0

    def GetAccLoss(self):
        losses = [t for t in self.trades if t[3] < 0]
        return min(losses) if losses else 0

    def GetMDD(self):
        max_drawdown = 0
        peak = -float('inf')
        for trade in self.trades:
            peak = max(peak, trade[3])
            drawdown = peak - trade[3]
            max_drawdown = max(max_drawdown, drawdown)
        return max_drawdown
def backtest_strategy(stock, long_ma_period, short_ma_period, move_stop_loss):
    stock['MA_long'] = stock['Close'].rolling(window=long_ma_period).mean()
    stock['MA_short'] = stock['Close'].rolling(window=short_ma_period).mean()

    order_record = OrderRecord()
    order_price = None
    stop_loss_point = None

    for i in range(1, len(stock)):
        if not pd.isna(stock['MA_long'].iloc[i-1]):
            if order_record.GetOpenInterest() == 0:
                if stock['MA_short'].iloc[i-1] <= stock['MA_long'].iloc[i-1] and stock['MA_short'].iloc[i] > stock['MA_long'].iloc[i]:
                    order_record.Order('Buy', stock['Date'].iloc[i], stock['Close'].iloc[i], 1)
                    order_price = stock['Close'].iloc[i]
                    stop_loss_point = order_price - move_stop_loss
                elif stock['MA_short'].iloc[i-1] >= stock['MA_long'].iloc[i-1] and stock['MA_short'].iloc[i] < stock['MA_long'].iloc[i]:
                    order_record.Order('Sell', stock['Date'].iloc[i], stock['Close'].iloc[i], 1)
                    order_price = stock['Close'].iloc[i]
                    stop_loss_point = order_price + move_stop_loss
            elif order_record.GetOpenInterest() == 1:
                if stock['Close'].iloc[i] - move_stop_loss > stop_loss_point:
                    stop_loss_point = stock['Close'].iloc[i] - move_stop_loss
                elif stock['Close'].iloc[i] < stop_loss_point:
                    order_record.Cover('Sell', stock['Date'].iloc[i], stock['Close'].iloc[i], 1)
            elif order_record.GetOpenInterest() == -1:
                if stock['Close'].iloc[i] + move_stop_loss < stop_loss_point:
                    stop_loss_point = stock['Close'].iloc[i] + move_stop_loss
                elif stock['Close'].iloc[i] > stop_loss_point:
                    order_record.Cover('Buy', stock['Date'].iloc[i], stock['Close'].iloc[i], 1)

    return order_record

def main():
    display_header()

    # 選擇資料區間
    st.subheader("選擇資料區間")
    start_date = st.date_input('選擇開始日期', datetime.date(2000, 1, 1), min_value=datetime.date(1900, 1, 1), max_value=datetime.date.today())
    end_date = st.date_input('選擇結束日期', datetime.date(2100, 12, 31), min_value=datetime.date(1900, 1, 1), max_value=datetime.date.today())
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

    # 輸入 SMA 和 EMA 的週期
    sma_period = st.number_input('請輸入SMA週期', min_value=1, max_value=100, value=20, step=1)
    ema_period = st.number_input('請輸入EMA週期', min_value=1, max_value=100, value=20, step=1)

    # 輸入布林通道的週期和標準差倍數
    bollinger_period = st.number_input('請輸入布林通道週期', min_value=1, max_value=100, value=20, step=1)
    bollinger_std = st.number_input('請輸入布林通道標準差倍數', min_value=0.1, max_value=10.0, value=2.0, step=0.1)

    # 輸入MACD的參數
    macd_short_period = st.number_input('請輸入MACD短期EMA週期', min_value=1, max_value=50, value=12, step=1)
    macd_long_period = st.number_input('請輸入MACD長期EMA週期', min_value=1, max_value=50, value=26, step=1)
    macd_signal_period = st.number_input('請輸入MACD信號線週期', min_value=1, max_value=50, value=9, step=1)

    # 輸入回測參數
    long_ma_period = st.number_input('請輸入長期MA週期', min_value=1, max_value=100, value=20, step=1)
    short_ma_period = st.number_input('請輸入短期MA週期', min_value=1, max_value=100, value=10, step=1)
    move_stop_loss = st.number_input('請輸入移動停損點數', min_value=0.1, max_value=100.0, value=30.0, step=0.1)

    # 驗證日期輸入
    if start_date > end_date:
        st.error("開始日期不能晚於結束日期")
    else:
        stock = load_stock_data(stockname, start_date, end_date, interval)
        if stock is not None:
            stock = calculate_indicators(stock, sma_period, ema_period, bollinger_period, bollinger_std, macd_short_period, macd_long_period, macd_signal_period)
            plot_stock_data(stock, sma_period, ema_period)

            # 執行回測
            order_record = backtest_strategy(stock, long_ma_period, short_ma_period, move_stop_loss)
            st.write("回測結果：")
            st.write("交易紀錄：", order_record.GetTradeRecord())
            st.write("淨利：", order_record.GetTotalProfit())
            st.write("勝率：", order_record.GetWinRate())
            st.write("最大連續虧損：", order_record.GetAccLoss())
            st.write("最大資金回落(MDD)：", order_record.GetMDD())

if __name__ == "__main__":
    main()

