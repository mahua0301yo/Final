###### 載入必要模組
import haohaninfo
from order_Lo8 import Record
import numpy as np
from talib.abstract import SMA,EMA, WMA, RSI, BBANDS, MACD
#import sys
import indicator_f_Lo2,datetime, indicator
import pandas as pd


###### 資料讀入與前處理
#df = pd.read_excel("kbars_台積電_1100701_1100708_2.xlsx")
#df = pd.read_excel("kbars_2330_2022-07-01-2022-07-31.xlsx") 
df = pd.read_excel("kbars_2330_2022-01-01-2024-04-09.xlsx")
df.columns  ## Index(['Unnamed: 0', 'time', 'open', 'low', 'high', 'close', 'volume','amount'], dtype='object')
df = df.drop('Unnamed: 0',axis=1)
df.columns  ## Index(['time', 'open', 'low', 'high', 'close', 'volume', 'amount'], dtype='object')
#df['time']
#type(df['time'])  ## pandas.core.series.Series
#df['time'][11]
df.head()



# ## 改變 KBar 時間長度 (參考以下連結: https://ithelp.ithome.com.tw/articles/10269151)
# # 作業:
# df.set_index( "time" , inplace=True) 
# df['time'] = df.index   

# kbars_5min_high = df.high.resample('5min').max()  #最高。以5分鐘重新取樣一次後，取最大值
# kbars_5min_low = df.low.resample('5min').min() #最低。以5分鐘重新取樣一次後，取最小值
# kbars_5min_close = df.close.resample('5min').last() #收盤。以5分鐘重新取樣一次後，取最後一筆
# kbars_5min_open = df.open.resample('5min').first() #開盤。以5分鐘重新取樣一次後，取第一筆
# kbars_5min_volume = df.volume.resample('5min').sum() #成交量，以5分鐘重新取樣一次後，取總和
# kbars_5min_amount = df.amount.resample('5min').sum() #金額，以5分鐘重新取樣一次後，取總和
# kbars_5min_product = df.product.resample('5min').last()
# kbars_5min_time = df.time.resample('5min').last()

# kbars_5min_high = df['high'].resample('5min', base=1).max()  #最高。以5分鐘重新取樣一次後，取最大值
# kbars_5min_low = df['low'].resample('5min', base=1).min() #最低。以5分鐘重新取樣一次後，取最小值
# kbars_5min_close = df['close'].resample('5min', base=1).last() #收盤。以5分鐘重新取樣一次後，取最後一筆
# kbars_5min_open = df['open'].resample('5min', base=1).first() #開盤。以5分鐘重新取樣一次後，取第一筆
# kbars_5min_volume = df['volume'].resample('5min', base=1).sum() #成交量，以5分鐘重新取樣一次後，取總和
# kbars_5min_amount = df['amount'].resample('5min', base=1).sum() #金額，以5分鐘重新取樣一次後，取總和
# #kbars_5min_product = str(df['product'].resample('5min').last())
# kbars_5min_product = df['product'].resample('5min', base=1).last()
# #kbars_5min_time = df['time'].resample('5min').last() 
# #kbars_5min_time = kbars_5min_time + pd.Series([1.0 for x in range(len(kbars_5min_time.index))])

# kbars_5min_high = df['high'].resample('5min').max()  #最高。以5分鐘重新取樣一次後，取最大值
# kbars_5min_low = df['low'].resample('5min').min() #最低。以5分鐘重新取樣一次後，取最小值
# kbars_5min_close = df['close'].resample('5min').last() #收盤。以5分鐘重新取樣一次後，取最後一筆
# kbars_5min_open = df['open'].resample('5min').first() #開盤。以5分鐘重新取樣一次後，取第一筆
# kbars_5min_volume = df['volume'].resample('5min').sum() #成交量，以5分鐘重新取樣一次後，取總和
# kbars_5min_amount = df['amount'].resample('5min').sum() #金額，以5分鐘重新取樣一次後，取總和
# #kbars_5min_product = str(df['product'].resample('5min').last())
# kbars_5min_product = df['product'].resample('5min').last()

# #df_5min_kbars = pd.concat([kbars_5min_time, kbars_5min_open, kbars_5min_low, kbars_5min_high, kbars_5min_close, kbars_5min_volume, kbars_5min_amount, kbars_5min_product], axis=1)
# df_5min_kbars = pd.concat([kbars_5min_open, kbars_5min_low, kbars_5min_high, kbars_5min_close, kbars_5min_volume, kbars_5min_amount, kbars_5min_product], axis=1)
# #df_5min_kbars['time'] = df_5min_kbars.index  
 
# df_5min_kbars['time'] = df_5min_kbars.index 
# df_5min_kbars.head()
# df = df_5min_kbars
# df.head()
# df['volume']

###### 畫 KBar 圖
# df.columns = [ i[0].upper()+i[1:] for i in df.columns ]
# df.set_index( "Time" , inplace=True)
# import mplfinance as mpf
# mpf.plot(df,volume=True,addplot=[],type='candle',style='charles')
df.set_index( "time" , inplace=True)
import mplfinance as mpf
mpf.plot(df,volume=True,addplot=[],type='candle',style='charles')

df['time'] = df.index

###### 轉化為字典:
KBar_dic = df.to_dict()
#type(KBar_dic)
#KBar_dic.keys()  ## dict_keys(['time', 'open', 'low', 'high', 'close', 'volume', 'amount'])
#KBar_dic['open']
#type(KBar_dic['open'])  ## dict
#KBar_dic['open'].values()
#type(KBar_dic['open'].values())  ## dict_values
KBar_open_list = list(KBar_dic['open'].values())
KBar_dic['open']=np.array(KBar_open_list)
#type(KBar_dic['open'])  ## numpy.ndarray
#KBar_dic['open'].shape  ## (1596,)
#KBar_dic['open'].size   ##  1596

KBar_dic['product'] = np.repeat('tsmc', KBar_dic['open'].size)
#KBar_dic['product'].size   ## 1596
#KBar_dic['product'][0]      ## 'tsmc'

KBar_time_list = list(KBar_dic['time'].values())
KBar_time_list = [i.to_pydatetime() for i in KBar_time_list] ## Timestamp to datetime
KBar_dic['time']=np.array(KBar_time_list)

# KBar_time_list[0]        ## Timestamp('2022-07-01 09:01:00')
# type(KBar_time_list[0])  ## pandas._libs.tslibs.timestamps.Timestamp
#KBar_time_list[0].to_pydatetime() ## datetime.datetime(2022, 7, 1, 9, 1)
#KBar_time_list[0].to_numpy()      ## numpy.datetime64('2022-07-01T09:01:00.000000000')
#KBar_dic['time']=np.array(KBar_time_list)
#KBar_dic['time'][80]   ## Timestamp('2022-09-01 23:02:00')

KBar_low_list = list(KBar_dic['low'].values())
KBar_dic['low']=np.array(KBar_low_list)

KBar_high_list = list(KBar_dic['high'].values())
KBar_dic['high']=np.array(KBar_high_list)

KBar_close_list = list(KBar_dic['close'].values())
KBar_dic['close']=np.array(KBar_close_list)

KBar_volume_list = list(KBar_dic['volume'].values())
KBar_dic['volume']=np.array(KBar_volume_list)

KBar_amount_list = list(KBar_dic['amount'].values())
KBar_dic['amount']=np.array(KBar_amount_list)


######  改變 KBar 時間長度   ########
# Product_array = np.array([])
# Time_array = np.array([])
# Open_array = np.array([])
# High_array = np.array([])
# Low_array = np.array([])
# Close_array = np.array([])
# Volume_array = np.array([])

Date = '20220101'
#KBar = indicator_f_Lo2.KBar(Date,'time',2)
KBar = indicator.KBar(Date,2880)

#KBar_dic['amount'].shape   ##(5585,)
#KBar_dic['amount'].size    ##5585
#KBar_dic['time'].size    ##5585

for i in range(KBar_dic['time'].size):
    
    #time = datetime.datetime.strptime(KBar_dic['time'][i],'%Y%m%d%H%M%S%f')
    time = KBar_dic['time'][i]
    #prod = KBar_dic['product'][i]
    price= KBar_dic['close'][i]
    qty =  KBar_dic['volume'][i]
    amount = KBar_dic['amount'][i]
    #tag=KBar.TimeAdd(time,price,qty,prod)
    tag=KBar.AddPrice(time,price,qty)
    
    # 更新K棒才判斷，若要逐筆判斷則 註解下面兩行, 因為計算 MA是利用收盤價, 而在 KBar class 中的 "TimeAdd"函數方法中, 收盤價只是一直附加最新的 price 而已.
    if tag != 1:
        continue
    #print(KBar.Time,KBar.GetOpen(),KBar.GetHigh(),KBar.GetLow(),KBar.GetClose(),KBar.GetVolume()) 
    
    
        
# #type(KBar.Time[1:-1]) ##numpy.ndarray       
# Time_array =  np.append(Time_array, KBar.Time[1:-1])    
# Open_array =  np.append(Open_array,KBar.Open[1:-1])
# High_array =  np.append(High_array,KBar.High[1:-1])
# Low_array =  np.append(Low_array,KBar.Low[1:-1])
# Close_array =  np.append(Close_array,KBar.Close[1:-1])
# Volume_array =  np.append(Volume_array,KBar.Volume[1:-1])
# Product_array = np.append(Product_array,KBar.Prod[1:-1])

# KBar_dic = {}

# ## 形成 KBar 字典:
# KBar_dic['time'] =  Time_array   
# KBar_dic['product'] =  Product_array
# KBar_dic['open'] =  Open_array
# KBar_dic['high'] =  High_array
# KBar_dic['low'] =  Low_array
# KBar_dic['close'] =  Close_array
# KBar_dic['volume'] =  Volume_array

 
###### 形成 變換長度後的KBar 字典:
KBar_dic = {}
KBar_dic['time'] =  KBar.TAKBar['time']   
#KBar_dic['product'] =  KBar.TAKBar['product']
KBar_dic['product'] = np.repeat('tsmc', KBar_dic['time'].size)
KBar_dic['open'] = KBar.TAKBar['open']
KBar_dic['high'] =  KBar.TAKBar['high']
KBar_dic['low'] =  KBar.TAKBar['low']
KBar_dic['close'] =  KBar.TAKBar['close']
KBar_dic['volume'] =  KBar.TAKBar['volume']

KBar_dic['time'].shape  ## (2814,)
KBar_dic['open'].shape  ## (2814,)
KBar_dic['high'].shape  ## (2814,)
KBar_dic['low'].shape  ## (2814,)
KBar_dic['close'].shape  ## (2814,)
KBar_dic['volume'].shape  ## (2814,)


###### 定義繪製相關圖形之函數
## 將K線轉為DataFrame
def KbarToDf(KBar_dic):
    # 將K線 Dictionary 轉換成 Dataframe
    Kbar_df=pd.DataFrame(KBar_dic)
    # 將 Dataframe 欄位名稱轉換
    Kbar_df.columns = [ i[0].upper()+i[1:] for i in Kbar_df.columns ]
    # 將 Time 欄位設為索引
    Kbar_df.set_index( "Time" , inplace=True)
    # 回傳
    return Kbar_df

## 繪製K線圖
def ChartKBar(KBar_dic,addp=[],volume_enable=True):
    # 將K線轉為DataFrame
    Kbar_df=KbarToDf(KBar_dic)
    # 開始繪圖
    mpf.plot(Kbar_df,volume=volume_enable,addplot=addp,type='candle',style='charles')

## 繪製K線圖以及下單紀錄
def ChartOrder(KBar_dic,TR,addp=[],volume_enable=True):
    # 將K線轉為DataFrame
    Kbar_df=KbarToDf(KBar_dic)
    ### 造 addp list
    # 買(多)方下單點位紀錄
    BTR = [ i for i in TR if i[0]=='Buy' or i[0]=='B' ]
    BuyOrderPoint = [] 
    BuyCoverPoint = []
    for date,value in Kbar_df['Close'].iteritems():
        # 買方進場
        if date in [ i[2] for i in BTR ]:
            BuyOrderPoint.append( Kbar_df['Low'][date] * 0.999 )
        else:
            BuyOrderPoint.append(np.nan)
        # 買方出場
        if date in [ i[4] for i in BTR ]:
            BuyCoverPoint.append( Kbar_df['High'][date] * 1.001 )
        else:
            BuyCoverPoint.append(np.nan)
    # 將下單點位加入副圖物件
    if [ i for i in BuyOrderPoint if not np.isnan(i) ] !=[]:
        addp.append(mpf.make_addplot(BuyOrderPoint,scatter=True,markersize=50,marker='^',color='red'))  ## 200
        addp.append(mpf.make_addplot(BuyCoverPoint,scatter=True,markersize=50,marker='v',color='blue')) ## 200
    # 賣(空)方下單點位紀錄
    STR = [ i for i in TR if i[0]=='Sell' or i[0]=='S' ]
    SellOrderPoint = [] 
    SellCoverPoint = []
    for date,value in Kbar_df['Close'].iteritems():
        # 賣方進場
        if date in [ i[2] for i in STR ]:
            SellOrderPoint.append( Kbar_df['High'][date] * 1.001 )
        else:
            SellOrderPoint.append(np.nan)
        # 賣方出場
        if date in [ i[4] for i in STR ]:
            SellCoverPoint.append( Kbar_df['Low'][date] * 0.999 )
        else:
            SellCoverPoint.append(np.nan)
    # 將下單點位加入副圖物件
    if [ i for i in SellOrderPoint if not np.isnan(i) ] !=[]:
        addp.append(mpf.make_addplot(SellOrderPoint,scatter=True,markersize=50,marker='v',color='green'))  ## 200
        addp.append(mpf.make_addplot(SellCoverPoint,scatter=True,markersize=50,marker='^',color='pink'))   ## 200
    # 開始繪圖 (將以上造的addp帶入以下)
    ChartKBar(KBar_dic,addp,volume_enable)  ## 多單進場: red向上 進, blue向下 出; 空單進場:green 向下進, pink 向上出

## 繪製K線圖加上MA以及下單紀錄
def ChartOrder_MA(KBar_dic,TR):
    # 將K線轉為DataFrame
    Kbar_df=KbarToDf(KBar_dic)
    # 定義指標副圖
    addp=[]
    addp.append(mpf.make_addplot(Kbar_df['MA_long'],color='red'))
    addp.append(mpf.make_addplot(Kbar_df['MA_short'],color='yellow'))
    # 繪製指標、下單圖
    ChartOrder(KBar_dic,TR,addp)




######  (一) 移動平均線策略   #########

##### 建立部位管理物件
OrderRecord=Record() 
# 取得回測參數、移動停損點數
# StartDate=sys.argv[1]
# EndDate=sys.argv[2]
# LongMAPeriod=int(sys.argv[3])
# ShortMAPeriod=int(sys.argv[4])
# MoveStopLoss=float(sys.argv[5])

#StartDate='20170330'
#EndDate='20170331'
LongMAPeriod=10
ShortMAPeriod=2
MoveStopLoss=10

##### 計算MA並添加至KBar_dic() 字典
KBar_dic['MA_long']=SMA(KBar_dic,timeperiod=LongMAPeriod)
KBar_dic['MA_short']=SMA(KBar_dic,timeperiod=ShortMAPeriod)


##### 開始回測
Order_Quantity=3
for n in range(0,len(KBar_dic['time'])-1):
    # 先判斷long MA的上一筆值是否為空值 再接續判斷策略內容
    if not np.isnan( KBar_dic['MA_long'][n-1] ) :
        ## 進場: 如果無未平倉部位
        #Order_Quantity=1
        if OrderRecord.GetOpenInterest()==0 :
            #Order_Quantity=1
            # 多單進場: 黃金交叉: short MA 向上突破 long MA
            if KBar_dic['MA_short'][n-1] <= KBar_dic['MA_long'][n-1] and KBar_dic['MA_short'][n] > KBar_dic['MA_long'][n] :
                #rder_Quantity=1
                OrderRecord.Order('Buy', KBar_dic['product'][n+1],KBar_dic['time'][n+1],KBar_dic['open'][n+1],Order_Quantity)
                OrderPrice = KBar_dic['open'][n+1]
                StopLossPoint = OrderPrice - MoveStopLoss
                continue
            # 空單進場:死亡交叉: short MA 向下突破 long MA
            if KBar_dic['MA_short'][n-1] >= KBar_dic['MA_long'][n-1] and KBar_dic['MA_short'][n] < KBar_dic['MA_long'][n] :
                #Order_Quantity=1
                OrderRecord.Order('Sell', KBar_dic['product'][n+1],KBar_dic['time'][n+1],KBar_dic['open'][n+1],Order_Quantity)
                OrderPrice = KBar_dic['open'][n+1]
                StopLossPoint = OrderPrice + MoveStopLoss
                continue
        ## 出場:
            
        ## 多單出場: 如果有多單部位   
        elif OrderRecord.GetOpenInterest()>0:
            ## 結算平倉(期貨才使用, 股票除非是下市櫃)
            if KBar_dic['product'][n+1] != KBar_dic['product'][n] :
                OrderRecord.Cover('Sell', KBar_dic['product'][n],KBar_dic['time'][n],KBar_dic['close'][n],OrderRecord.GetOpenInterest())
                continue
            # 更新停損價位 (移動停損點)
            if KBar_dic['close'][n] - MoveStopLoss > StopLossPoint :
                StopLossPoint = KBar_dic['close'][n] - MoveStopLoss
            # 如果上一根K的收盤價觸及停損價位，則在最新時間出場
            elif KBar_dic['close'][n] < StopLossPoint :
                OrderRecord.Cover('Sell', KBar_dic['product'][n+1],KBar_dic['time'][n+1],KBar_dic['open'][n+1],OrderRecord.GetOpenInterest())
                continue
        # 空單出場: 如果有空單部位
        elif OrderRecord.GetOpenInterest()<0:
            ## 結算平倉(期貨才使用, 股票除非是下市櫃)
            if KBar_dic['product'][n+1] != KBar_dic['product'][n] :
           
                OrderRecord.Cover('Buy', KBar_dic['product'][n],KBar_dic['time'][n],KBar_dic['close'][n],-OrderRecord.GetOpenInterest())
                continue
            # 更新停損價位 (移動停損)
            if KBar_dic['close'][n] + MoveStopLoss < StopLossPoint :
                StopLossPoint = KBar_dic['close'][n] + MoveStopLoss
            # 如果上一根K的收盤價觸及停損價位，則在最新時間出場
            elif KBar_dic['close'][n] > StopLossPoint :
                OrderRecord.Cover('Buy', KBar_dic['product'][n+1],KBar_dic['time'][n+1],KBar_dic['open'][n+1],-OrderRecord.GetOpenInterest())
                continue
                
##### 繪製走勢圖加上MA以及下單點位
#from chart import ChartOrder_MA
ChartOrder_MA(KBar_dic,OrderRecord.GetTradeRecord())
KBar_dic.keys()

##### 計算績效:
OrderRecord.GetTradeRecord()  ## 交易紀錄清單
OrderRecord.GetProfit()       ## 利潤清單
OrderRecord.GetTotalProfit()  ## 淨利
OrderRecord.GetWinRate()      ## 勝率
OrderRecord.GetAccLoss()      ## 最大連續虧損
OrderRecord.GetMDD()          ## 最大資金回落(MDD)



###### (二) RSI 順勢策略   
# 建立部位管理物件
OrderRecord=Record() 
# 取得回測參數、移動停損點數
#StartDate=sys.argv[1]
#EndDate=sys.argv[2]
LongRSIPeriod=10
ShortRSIPeriod=5
MoveStopLoss=30
# 回測取報價物件
#KBar=GOrder.GetTAKBar(StartDate,EndDate,'TXF','Future','0','10')
# 計算 RSI指標長短線, 以及定義中線
KBar_dic['RSI_long']=RSI(KBar_dic,timeperiod=LongRSIPeriod)
KBar_dic['RSI_short']=RSI(KBar_dic,timeperiod=ShortRSIPeriod)
KBar_dic['Middle']=np.array([50]*len(KBar_dic['time']))



##### 開始回測
for n in range(1,len(KBar_dic['time'])-1):
    # 先判斷long MA的上一筆值是否為空值 再接續判斷策略內容
    if not np.isnan( KBar_dic['RSI_long'][n-1] ) :
        # 如果無未平倉部位
        if OrderRecord.GetOpenInterest()==0 :
            # 黃金交叉:short RSI 從下往上穿越 long RSI 並且 long RSI 大於 50(順勢: 漲)
            if KBar_dic['RSI_short'][n-1] <= KBar_dic['RSI_long'][n-1] and KBar_dic['RSI_short'][n] > KBar_dic['RSI_long'][n] and KBar_dic['RSI_long'][n] > KBar_dic['Middle'][n] :
                OrderRecord.Order('Buy', KBar_dic['product'][n+1],KBar_dic['time'][n+1],KBar_dic['open'][n+1],1)
                OrderPrice = KBar_dic['open'][n+1]
                StopLossPoint = OrderPrice - MoveStopLoss
                #print(KBar_dic['time'][n] ,'Buy',KBar_dic['RSI_short'][n-1] ,KBar_dic['RSI_long'][n-1] , KBar_dic['RSI_short'][n], KBar_dic['RSI_long'][n])
                continue
            # 死亡交叉: short RSI 從上往下穿越 long RSI 並且 long RSI 小於 50(順勢:跌)
            if KBar_dic['RSI_short'][n-1] >= KBar_dic['RSI_long'][n-1] and KBar_dic['RSI_short'][n] < KBar_dic['RSI_long'][n] and KBar_dic['RSI_long'][n] < KBar_dic['Middle'][n] :
                OrderRecord.Order('Sell', KBar_dic['product'][n+1],KBar_dic['time'][n+1],KBar_dic['open'][n+1],1)
                OrderPrice = KBar_dic['open'][n+1]
                StopLossPoint = OrderPrice + MoveStopLoss
                continue
        # 如果有多單部位   
        elif OrderRecord.GetOpenInterest()==1 :
            # 結算平倉
            if KBar_dic['product'][n+1] != KBar_dic['product'][n] :
                OrderRecord.Cover('Sell', KBar_dic['product'][n],KBar_dic['time'][n],KBar_dic['close'][n],1)
                continue
            # 逐筆更新移動停損價位
            if KBar_dic['close'][n] - MoveStopLoss > StopLossPoint :
                StopLossPoint = KBar_dic['close'][n] - MoveStopLoss
            # 如果上一根K的收盤價觸及停損價位，則在最新時間出場
            elif KBar_dic['close'][n] < StopLossPoint :
                OrderRecord.Cover('Sell', KBar_dic['product'][n+1],KBar_dic['time'][n+1],KBar_dic['open'][n+1],1)
                continue
        # 如果有空單部位
        elif OrderRecord.GetOpenInterest()==-1 :
            # 結算平倉
            if KBar_dic['product'][n+1] != KBar_dic['product'][n] :
                OrderRecord.Cover('Buy', KBar_dic['product'][n],KBar_dic['time'][n],KBar_dic['close'][n],1)
                continue
            # 逐筆更新移動停損價位
            if KBar_dic['close'][n] + MoveStopLoss < StopLossPoint :
                StopLossPoint = KBar_dic['close'][n] + MoveStopLoss
            # 如果上一根K的收盤價觸及停損價位，則在最新時間出場
            elif KBar_dic['close'][n] > StopLossPoint :
                OrderRecord.Cover('Buy', KBar_dic['product'][n+1],KBar_dic['time'][n+1],KBar_dic['open'][n+1],1)
                continue

##### 計算與查看績效                
## 將回測紀錄寫至MicroTest中
#OrderRecord.FutureMicroTestRecord('5-3-4-F',200,50)
OrderRecord.GetTradeRecord()  ## 交易紀錄清單
OrderRecord.GetProfit()       ## 利潤清單
OrderRecord.GetTotalProfit()  ## 淨利
OrderRecord.GetWinRate()      ## 勝率
OrderRecord.GetAccLoss()      ## 最大連續虧損
OrderRecord.GetMDD()          ## 最大資金回落(MDD)
OrderRecord.GeneratorProfitChart(StrategyName='RSI-long_short_cross')

##### 繪製走勢圖加上RSI以及下單點位
from chart import ChartOrder_RSI_1, ChartKBar_RSI_1
ChartKBar_RSI_1(KBar_dic,LongRSIPeriod,ShortRSIPeriod)
ChartOrder_RSI_1(KBar_dic,OrderRecord.GetTradeRecord())




###### (三) RSI 逆勢策略   #########
# 建立部位管理物件
OrderRecord=Record() 
##計算 RSI指標, 天花板與地板
RSIPeriod=5
Ceil=80
Floor=20
MoveStopLoss=30
KBar_dic['RSI']=RSI(KBar_dic,timeperiod=RSIPeriod)
KBar_dic['Ceil']=np.array([Ceil]*len(KBar_dic['time']))
KBar_dic['Floor']=np.array([Floor]*len(KBar_dic['time']))


##### 開始回測
for n in range(1,len(KBar_dic['time'])-1):
    # 先判斷long MA的上一筆值是否為空值 再接續判斷策略內容
    if not np.isnan( KBar_dic['RSI'][n-1] ) :
        # 如果無未平倉部位
        if OrderRecord.GetOpenInterest()==0 :
            # RSI 向上突破超賣界線:
            if KBar_dic['RSI'][n-1] <= KBar_dic['Floor'][n-1] and KBar_dic['RSI'][n] > KBar_dic['Floor'][n]:
                OrderRecord.Order('Buy', KBar_dic['product'][n+1],KBar_dic['time'][n+1],KBar_dic['open'][n+1],1)
                OrderPrice = KBar_dic['open'][n+1]
                StopLossPoint = OrderPrice - MoveStopLoss
                continue
            # RSI 向下突破超買界線:
            if KBar_dic['RSI'][n-1] >= KBar_dic['Ceil'][n-1] and KBar_dic['RSI'][n] < KBar_dic['Ceil'][n] :
                OrderRecord.Order('Sell', KBar_dic['product'][n+1],KBar_dic['time'][n+1],KBar_dic['open'][n+1],1)
                OrderPrice = KBar_dic['open'][n+1]
                StopLossPoint = OrderPrice + MoveStopLoss
                continue
        # 如果有多單部位   
        elif OrderRecord.GetOpenInterest()==1 :
            # 結算平倉
            if KBar_dic['product'][n+1] != KBar_dic['product'][n] :
                OrderRecord.Cover('Sell', KBar_dic['product'][n],KBar_dic['time'][n],KBar_dic['close'][n],1)
                continue
            # 逐筆更新移動停損價位
            if KBar_dic['close'][n] - MoveStopLoss > StopLossPoint :
                StopLossPoint = KBar_dic['close'][n] - MoveStopLoss
            # 如果上一根K的收盤價觸及停損價位，則在最新時間出場
            elif KBar_dic['close'][n] < StopLossPoint :
                OrderRecord.Cover('Sell', KBar_dic['product'][n+1],KBar_dic['time'][n+1],KBar_dic['open'][n+1],1)
                continue
            # 若 RSI 大於 超買界線 則停利出場
            if KBar_dic['RSI'][n] > KBar_dic['Ceil'][n]:
                OrderRecord.Cover('Sell', KBar_dic['product'][n+1],KBar_dic['time'][n+1],KBar_dic['open'][n+1],1)
                continue
        # 如果有空單部位
        elif OrderRecord.GetOpenInterest()==-1 :
            # 結算平倉
            if KBar_dic['product'][n+1] != KBar_dic['product'][n] :
                OrderRecord.Cover('Buy', KBar_dic['product'][n],KBar_dic['time'][n],KBar_dic['close'][n],1)
                continue
            # 逐筆更新移動停損價位
            if KBar_dic['close'][n] + MoveStopLoss < StopLossPoint :
                StopLossPoint = KBar_dic['close'][n] + MoveStopLoss
            # 如果上一根K的收盤價觸及停損價位，則在最新時間出場
            elif KBar_dic['close'][n] > StopLossPoint :
                OrderRecord.Cover('Buy', KBar_dic['product'][n+1],KBar_dic['time'][n+1],KBar_dic['open'][n+1],1)
                continue
            # 若 RSI 低於 超賣界線 則停利出場
            if KBar_dic['RSI'][n] < KBar_dic['Floor'][n]:
                OrderRecord.Cover('Buy', KBar_dic['product'][n+1],KBar_dic['time'][n+1],KBar_dic['open'][n+1],1)
                continue
                
##### 計算與查看績效 
# 將回測紀錄寫至MicroTest中
#OrderRecord.FutureMicroTestRecord('5-3-5-F',200,50)
#OrderRecord.FutureMicroTestRecord('5-3-5-F',200,50,1,'pu25','pu25')
## 查看MicroTest 績效網頁連結: http://140.128.36.207/exchange1/MicroTest/index.php
OrderRecord.GetTradeRecord()  ## 交易紀錄清單
OrderRecord.GetProfit()       ## 利潤清單
OrderRecord.GetTotalProfit()  ## 淨利
OrderRecord.GetWinRate()      ## 勝率
OrderRecord.GetAccLoss()      ## 最大連續虧損
OrderRecord.GetMDD()          ## 最大資金回落(MDD)
OrderRecord.GeneratorProfitChart(StrategyName='RSI-long_short_cross')

##### 繪製走勢圖加上MA以及下單點位
from chart import ChartOrder_RSI_2, ChartKBar_RSI_2
ChartKBar_RSI_2(KBar_dic,RSIPeriod,Ceil, Floor)
ChartOrder_RSI_2(KBar_dic,OrderRecord.GetTradeRecord()) 


###### (四) 布林通道 策略   
#from talib.abstract import BBANDS, MACD
# 建立部位管理物件
OrderRecord=Record() 
# 取得回測參數、移動停損點數
BBANDSPeriod=60
MoveStopLoss=30
KBar_dic['Upper'],KBar_dic['Middle'],KBar_dic['Lower']=BBANDS(KBar_dic,timeperiod=BBANDSPeriod)

##### 開始回測 (布林通道部分)
for n in range(1,len(KBar_dic['time'])-1):
    # 先判斷中間線的上一筆值是否為空值 再接續判斷策略內容
    if not np.isnan( KBar_dic['Middle'][n-1] ) :
        # 如果無未平倉部位
        if OrderRecord.GetOpenInterest()==0 :
            # 當價格由下向上穿越下界線 則進場做多
            if KBar_dic['close'][n-1] <= KBar_dic['Lower'][n-1] and KBar_dic['close'][n] > KBar_dic['Lower'][n]:
                OrderRecord.Order('Buy', KBar_dic['product'][n+1],KBar_dic['time'][n+1],KBar_dic['open'][n+1],1)
                OrderPrice = KBar_dic['open'][n+1]
                StopLossPoint = OrderPrice - MoveStopLoss
                continue
            # 當價格由上向下穿越上界線 則進場做空
            if KBar_dic['close'][n-1] >= KBar_dic['Upper'][n-1] and KBar_dic['close'][n] < KBar_dic['Upper'][n]:
                OrderRecord.Order('Sell', KBar_dic['product'][n+1],KBar_dic['time'][n+1],KBar_dic['open'][n+1],1)
                OrderPrice = KBar_dic['open'][n+1]
                StopLossPoint = OrderPrice + MoveStopLoss
                continue
        # 如果有多單部位   
        elif OrderRecord.GetOpenInterest()==1 :
            # 結算平倉
            if KBar_dic['product'][n+1] != KBar_dic['product'][n] :
                OrderRecord.Cover('Sell', KBar_dic['product'][n],KBar_dic['time'][n],KBar_dic['close'][n],1)
                continue
            # 逐筆更新移動停損價位
            if KBar_dic['close'][n] - MoveStopLoss > StopLossPoint :
                StopLossPoint = KBar_dic['close'][n] - MoveStopLoss
            # 如果上一根K的收盤價觸及停損價位，則在最新時間出場
            elif KBar_dic['close'][n] < StopLossPoint :
                OrderRecord.Cover('Sell', KBar_dic['product'][n+1],KBar_dic['time'][n+1],KBar_dic['open'][n+1],1)
                continue
            # 若大於 布林通道上界 則停利出場
            if KBar_dic['close'][n] >= KBar_dic['Upper'][n]:
                OrderRecord.Cover('Sell', KBar_dic['product'][n+1],KBar_dic['time'][n+1],KBar_dic['open'][n+1],1)
                continue
        # 如果有空單部位
        elif OrderRecord.GetOpenInterest()==-1 :
            # 結算平倉
            if KBar_dic['product'][n+1] != KBar_dic['product'][n] :
                OrderRecord.Cover('Buy', KBar_dic['product'][n],KBar_dic['time'][n],KBar_dic['close'][n],1)
                continue
            # 逐筆更新移動停損價位
            if KBar_dic['close'][n] + MoveStopLoss < StopLossPoint :
                StopLossPoint = KBar_dic['close'][n] + MoveStopLoss
            # 如果上一根K的收盤價觸及停損價位，則在最新時間出場
            elif KBar_dic['close'][n] > StopLossPoint :
                OrderRecord.Cover('Buy', KBar_dic['product'][n+1],KBar_dic['time'][n+1],KBar_dic['open'][n+1],1)
                continue
            # 若小於 布林通道下界 則停利出場
            if KBar_dic['close'][n] <= KBar_dic['Lower'][n]:
                OrderRecord.Cover('Buy', KBar_dic['product'][n+1],KBar_dic['time'][n+1],KBar_dic['open'][n+1],1)
                continue
                
##### 計算與查看績效 
# 將回測紀錄寫至MicroTest中
#OrderRecord.FutureMicroTestRecord('5-3-5-F',200,50)
#OrderRecord.FutureMicroTestRecord('5-3-5-F',200,50,1,'pu25','pu25')
## 查看MicroTest 績效網頁連結: http://140.128.36.207/exchange1/MicroTest/index.php
OrderRecord.GetTradeRecord()  ## 交易紀錄清單
OrderRecord.GetProfit()       ## 利潤清單
OrderRecord.GetTotalProfit()  ## 淨利
OrderRecord.GetWinRate()      ## 勝率
OrderRecord.GetAccLoss()      ## 最大連續虧損
OrderRecord.GetMDD()          ## 最大資金回落(MDD)
OrderRecord.GeneratorProfitChart(StrategyName='RSI-long_short_cross')


##### 繪製走勢圖加上MA以及下單點位
from chart import ChartOrder_BBANDS
ChartOrder_BBANDS(KBar_dic,OrderRecord.GetTradeRecord())


###### (五) MACD 策略   
## 對於日K, 以下的三個數字時常設定為(依序): 12, 26, 9
Fastperiod = 15
Slowperiod = 20
Signalperiod = 9

## macd="DIF", macdsignal="DEA", macdhist="DIF-DEA"="MACD"(另本書的符號)
KBar_dic['macd'],KBar_dic['macdsignal'],KBar_dic['macdhist']=MACD(KBar_dic,fastperiod=Fastperiod, slowperiod=Slowperiod, signalperiod=Signalperiod)

##### 策略
## 1. 多方進場: macdhist>0; 多方出場: macdhist<0
## 2. 多方進場: macdhist>0 且 macdsignal>0
## 3. 多方進場: macd>0 且 macdsignal>0 且 (macd 向上突破 macdsignal(黃金交叉))
## 4. 空方進場: macd<0 且 macdsignal<0 且 (macd 向下突破 macdsignal(死亡交叉))










# KBar_dic['product'][n+1]  ## 'tsmc'
# len(KBar_dic['product'])  ## 1596
# type(KBar_dic)  ##dict
# type(OrderRecord.GetTradeRecord())  ## list
# len(OrderRecord.GetTradeRecord()) ##1


### Pandas DataFrame iteritems() Method:
import pandas as pd
data = {
  "firstname": ["Sally", "Mary", "John"],
  "age": [50, 40, 30]
}
df = pd.DataFrame(data)
df.head()

for x, y in df["firstname"].iteritems():
  print(x)
  print(y)
  
for x, y in df["age"].iteritems():
  print(x)
  print(y)