import pandas as pd
import numpy as np
import talib as ta
from matplotlib import pyplot as plt
import seaborn as sns

path = ".\\Dataset\\"
filename = "Company.NS"
df = pd.read_csv(path + filename + ".csv")

''' Correct Null Entries '''
df.Date = pd.to_datetime(df.Date)
df_null = df[df.Open.isnull()]
indices = df_null.index.tolist()
for i in indices:
    date = df["Date"].loc[i]
    df.loc[i] = df.loc[i-1]
    df["Date"].loc[i] = date
df = df.reset_index(drop=True)
columns = df.columns[1:5]

span = 30

''' Calculate Features '''

df["Mean"] = pd.DataFrame.mean(df[columns],axis=1)
# df["return"] = np.nan

''' Simple Moving Average '''
# df["Mov_Avg"] = np.nan
# df["SMA"] = df["Close"].rolling(window = span, min_periods= span).mean()
df['SMA'] = ta.SMA(df['Close'], timeperiod=span)

''' Exponential Moving Average '''
# df["Exp_MOv_Avg"] = np.nan
# df["EMA"] = df["Close"][span-1:].ewm(span = span, adjust = False).mean()
df["EMA"] = ta.EMA(df['Close'], timeperiod=span)

''' Aroon Osc '''
# df["Aroon Osc"] = np.nan
df['Aroon Osc'] = ta.AROONOSC(df['High'], df['Low'], timeperiod=span)

''' MACD '''
# macd, macdsignal, macdhist = ta.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
# df['MACD'] = macdsignal

''' RSI '''
# df["RSI"] = np.nan
# df["RSI"] = rsiFunc(df["Adj Close"].values.tolist())
df['RSI'] = ta.RSI(np.array(df['Close']), timeperiod=span)

''' BBands '''
upperband, middleband, lowerband = ta.BBANDS(df['Close'], timeperiod=span, nbdevup=2, nbdevdn=2, matype=0)
df['BBand'] = middleband

''' Stochastic Momentum Indicator '''
# slowk, slowd = ta.STOCH(df['High'], df['Low'], df['Close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
# df['StochMI'] = [i/j for i,j in zip(slowk,slowd)]

''' Chande Momentum Oscillator '''
df['CMO'] = ta.CMO(df['Close'], timeperiod=span)

''' Commodity Channel Index '''
df['CCI'] = ta.CMO(df['Close'], timeperiod=span)

''' Rate of Price Change '''
df['ROC_Close'] = ta.ROC(df['Close'], timeperiod=span)

''' Rate of Volume Change '''
df['ROC_Vol'] = ta.ROC(df['Volume'], timeperiod=span)

''' William %R '''
# df["William%R"] = np.nan
df["WillR"] = ta.WILLR(df['High'], df['Low'], df['Close'], timeperiod=span)

''' *SAR '''
# df['SAR'] = ta.SAR(np.array(df['High']), np.array(df['Low']),0.2,0.2)

''' *ADX '''
# df['ADX'] = ta.ADX(np.array(df['High']), np.array(df['Low']),np.array(df['Close']), timeperiod = span)

''' *Correlation '''
# df['Corr'] = df['SMA'].rolling(window=span).corr(df['Close'])

''' Target '''
df['Target'] = df['Close'].shift(-span)

pd.DataFrame.to_csv(df[span:-span],path + filename+"_Features.csv",index=False)