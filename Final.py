# 驗證日期輸入
if start_date > end_date:
    st.error("開始日期不能晚於結束日期")
else:
    try:
        # 讀取股票資料
        stock = yf.download(stockname, start=start_date, end=end_date)
        
        if stock.empty:
            st.error("未能讀取到數據，請檢查股票代號是否正確")
        else:
            st.success("數據讀取成功")

            # 將 Volume 改為 amount
            stock.rename(columns={'Volume': 'amount'}, inplace=True)
            
            # 使用 amount / Open 計算新的 Volume
            stock['Volume'] = stock['amount'] / stock['Open']
            
            # 互換 Volume 和 amount 兩欄的位置
            columns = stock.columns.tolist()
            vol_idx = columns.index('Volume')
            amt_idx = columns.index('amount')
            columns[vol_idx], columns[amt_idx] = columns[amt_idx], columns[vol_idx]
            stock = stock[columns]
            
            # 顯示股票數據
            st.write(stock)

            # 轉化為字典
            KBar_dic = stock.to_dict()
            KBar_dic['product'] = np.repeat(stockname, len(stock))
            
            # 進一步數據處理和可視化
            # ...
    
    except Exception as e:
        st.error(f"讀取數據時出錯: {e}")
