import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go

# Display the header
def display_header():
    st.title("股票技術指標交易策略")
    st.write("這是一個基於Streamlit的應用，用於展示股票的技術指標和相應的交易策略。")

# Load stock data
def load_stock_data(stockname, start_date, end_date, interval):
    try:
        stock_data = yf.download(stockname, start=start_date, end=end_date, interval=interval)
        stock_data.reset_index(inplace=True)
        return stock_data
    except Exception as e:
        st.error(f"無法載入資料: {e}")
        return None

# Calculate technical indicators
def calculate_indicators(stock, bollinger_params, macd_params):
    # KDJ calculation
    low_min = stock['Low'].rolling(window=9).min()
    high_max = stock['High'].rolling(window=9).max()
    stock['RSV'] = 100 * (stock['Close'] - low_min) / (high_max - low_min)
    stock['K'] = stock['RSV'].ewm(com=2).mean()
    stock['D'] = stock['K'].ewm(com=2).mean()
    stock['J'] = 3 * stock['K'] - 2 * stock['D']

    # Bollinger Bands calculation
    period, multiplier = bollinger_params
    stock['Middle_Band'] = stock['Close'].rolling(window=period).mean()
    stock['STD'] = stock['Close'].rolling(window=period).std()
    stock['Upper_Band'] = stock['Middle_Band'] + (stock['STD'] * multiplier)
    stock['Lower_Band'] = stock['Middle_Band'] - (stock['STD'] * multiplier)

    # MACD calculation
    short_period, long_period, signal_period = macd_params
    stock['EMA12'] = stock['Close'].ewm(span=short_period, adjust=False).mean()
    stock['EMA26'] = stock['Close'].ewm(span=long_period, adjust=False).mean()
    stock['MACD'] = stock['EMA12'] - stock['EMA26']
    stock['Signal_Line'] = stock['MACD'].ewm(span=signal_period, adjust=False).mean()

    return stock

# KDJ strategy
def kdj_strategy(stock):
    order_record = OrderRecord()
    for n in range(1, len(stock)):
        if stock['K'][n-1] < stock['D'][n-1] and stock['K'][n] > stock['D'][n]:
            order_record.Order('Buy', 'KDJ', stock['Date'][n], stock['Open'][n], 1)
        elif stock['K'][n-1] > stock['D'][n-1] and stock['K'][n] < stock['D'][n]:
            order_record.Cover('Sell', 'KDJ', stock['Date'][n], stock['Open'][n], 1)
    return order_record

# Bollinger Bands strategy
def bollinger_strategy(stock):
    order_record = OrderRecord()
    for n in range(1, len(stock)):
        if stock['Close'][n-1] <= stock['Upper_Band'][n-1] and stock['Close'][n] > stock['Upper_Band'][n]:
            order_record.Cover('Sell', 'Bollinger', stock['Date'][n], stock['Open'][n], 1)
        elif stock['Close'][n-1] >= stock['Lower_Band'][n-1] and stock['Close'][n] < stock['Lower_Band'][n]:
            order_record.Order('Buy', 'Bollinger', stock['Date'][n], stock['Open'][n], 1)
    return order_record

# MACD strategy
def macd_strategy(stock):
    order_record = OrderRecord()
    for n in range(1, len(stock)):
        if stock['MACD'][n-1] < stock['Signal_Line'][n-1] and stock['MACD'][n] > stock['Signal_Line'][n]:
            order_record.Order('Buy', 'MACD', stock['Date'][n], stock['Open'][n], 1)
        elif stock['MACD'][n-1] > stock['Signal_Line'][n-1] and stock['MACD'][n] < stock['Signal_Line'][n]:
            order_record.Cover('Sell', 'MACD', stock['Date'][n], stock['Open'][n], 1)
    return order_record

# Plot trade points
def plot_trade_points(fig, trades):
    for trade in trades:
        color = 'green' if trade['action'] == 'Buy' else 'red'
        symbol = 'arrow-up' if trade['action'] == 'Buy' else 'arrow-down'
        fig.add_trace(go.Scatter(
            x=[trade['time']],
            y=[trade['price']],
            mode='markers',
            marker=dict(color=color, size=15, symbol=symbol),
            name=f"{trade['product']} {trade['action']}"
        ))

# Main function
def main():
    display_header()

    st.subheader("選擇資料區間")
    start_date = st.date_input('選擇開始日期', datetime.date(2015, 1, 1))
    end_date = st.date_input('選擇結束日期', datetime.date(2100, 12, 31))
    stockname = st.text_input('請輸入股票代號 (例: 2330.TW)', '2330.TW')

    interval_options = {"1天": "1d", "1星期": "1wk", "1個月": "1mo", "3個月": "3mo"}
    interval_label = st.selectbox("選擇K線時間長", list(interval_options.keys()))
    interval = interval_options[interval_label]

    bollinger_params = (
        st.number_input('布林通道週期', min_value=1, max_value=100, value=20, step=1),
        st.number_input('布林通道標準差倍數', min_value=0.1, max_value=10.0, value=2.0, step=0.1)
    )
    macd_params = (
        st.number_input('MACD短期EMA週期', min_value=1, max_value=50, value=12, step=1),
        st.number_input('MACD長期EMA週期', min_value=1, max_value=50, value=26, step=1),
        st.number_input('MACD信號線週期', min_value=1, max_value=50, value=9, step=1)
    )

    if start_date > end_date:
        st.error("開始日期不能晚於結束日期")
    else:
        stock = load_stock_data(stockname, start_date, end_date, interval)
        if stock is not None:
            stock = calculate_indicators(stock, bollinger_params, macd_params)
            kdj_record = kdj_strategy(stock)
            bollinger_record = bollinger_strategy(stock)
            macd_record = macd_strategy(stock)
            plot_stock_data(stock, kdj_record, bollinger_record, macd_record)

# Plot stock data with indicators and trade points
def plot_stock_data(stock, kdj_record, bollinger_record, macd_record):
    fig_kline = go.Figure()
    fig_kline.add_trace(go.Candlestick(
        x=stock['Date'], open=stock['Open'], high=stock['High'], low=stock['Low'], close=stock['Close'], name='價格'))
    fig_kline.add_trace(go.Scatter(x=stock['Date'], y=stock['Middle_Band'], line=dict(color='blue', width=1), name='中軌'))
    fig_kline.add_trace(go.Scatter(x=stock['Date'], y=stock['Upper_Band'], line=dict(color='red', width=1), name='上軌'))
    fig_kline.add_trace(go.Scatter(x=stock['Date'], y=stock['Lower_Band'], line=dict(color='red', width=1), name='下軌'))

    plot_trade_points(fig_kline, kdj_record.GetTradeRecord())
    plot_trade_points(fig_kline, bollinger_record.GetTradeRecord())
    plot_trade_points(fig_kline, macd_record.GetTradeRecord())

    fig_kline.update_layout(title='K線圖與技術指標', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_kline, use_container_width=True)

    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=stock['Date'], y=stock['MACD'], line=dict(color='blue', width=1), name='MACD'))
    fig_macd.add_trace(go.Scatter(x=stock['Date'], y=stock['Signal_Line'], line=dict(color='orange', width=1), name='信號線'))
    fig_macd.add_trace(go.Bar(x=stock['Date'], y=stock['MACD'] - stock['Signal_Line'], name='柱狀圖'))

    fig_macd.update_layout(title='MACD', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_macd, use_container_width=True)

    fig_kdj = go.Figure()
    fig_kdj.add_trace(go.Scatter(x=stock['Date'], y=stock['K'], line=dict(color='blue', width=1), name='K'))
    fig_kdj.add_trace(go.Scatter(x=stock['Date'], y=stock['D'], line=dict(color='orange', width=1), name='D'))
    fig_kdj.add_trace(go.Scatter(x=stock['Date'], y=stock['J'], line=dict(color='green', width=1), name='J'))

    fig_kdj.update_layout(title='KDJ', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_kdj, use_container_width=True)

    st.subheader("交易績效")
    st.write("KDJ策略")
    kdj_trade_record = kdj_record.GetTradeRecord()
    kdj_profit = kdj_record.GetProfit()
    kdj_total_profit = kdj_record.GetTotalProfit()
    kdj_win_rate = kdj_record.GetWinRate()
    kdj_acc_loss = kdj_record.GetAccLoss()
    kdj_mdd = kdj_record.GetMDD()
    st.write(f"交易紀錄: {kdj_trade_record}")
    st.write(f"損益: {kdj_profit}")
    st.write(f"總損益: {kdj_total_profit}")
    st.write(f"勝率: {kdj_win_rate * 100}%")
    st.write(f"累積損失: {kdj_acc_loss}")
    st.write(f"最大回撤: {kdj_mdd}")

    st.write("布林通道策略")
    bollinger_trade_record = bollinger_record.GetTradeRecord()
    bollinger_profit = bollinger_record.GetProfit()
    bollinger_total_profit = bollinger_record.GetTotalProfit()
    bollinger_win_rate = bollinger_record.GetWinRate()
    bollinger_acc_loss = bollinger_record.GetAccLoss()
    bollinger_mdd = bollinger_record.GetMDD()
    st.write(f"交易紀錄: {bollinger_trade_record}")
    st.write(f"損益: {bollinger_profit}")
    st.write(f"總損益: {bollinger_total_profit}")
    st.write(f"勝率: {bollinger_win_rate * 100}%")
    st.write(f"累積損失: {bollinger_acc_loss}")
    st.write(f"最大回撤: {bollinger_mdd}")

    st.write("MACD策略")
    macd_trade_record = macd_record.GetTradeRecord()
    macd_profit = macd_record.GetProfit()
    macd_total_profit = macd_record.GetTotalProfit()
    macd_win_rate = macd_record.GetWinRate()
    macd_acc_loss = macd_record.GetAccLoss()
    macd_mdd = macd_record.GetMDD()
    st.write(f"交易紀錄: {macd_trade_record}")
    st.write(f"損益: {macd_profit}")
    st.write(f"總損益: {macd_total_profit}")
    st.write(f"勝率: {macd_win_rate * 100}%")
    st.write(f"累積損失: {macd_acc_loss}")
    st.write(f"最大回撤: {macd_mdd}")

if __name__ == "__main__":
    main()
