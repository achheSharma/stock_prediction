import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

filename = "AXISBANK.NS"
df = pd.read_csv(filename + ".csv")
# rng = pd.date_range(start=df["Date"].loc[0], end=df["Date"].loc[len(df)-1], freq='B')
# print(len(rng))
df.Date = pd.to_datetime(df.Date)
df_null = df[df.Open.isnull()]
indices = df_null.index.tolist()
for i in indices:
    # print(i.date())
    date = df["Date"].loc[i]
    df.loc[i] = df.loc[i-1]
    df["Date"].loc[i] = date
# df = df.dropna()
df = df.reset_index(drop=True)
columns = df.columns[1:5]

span = 30

def rsiFunc(prices, n=span):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi


df["Mean"]= pd.DataFrame.mean(df[columns],axis=1)
# df["return"] = np.nan

''' Simple Moving Average '''
# df["Mov_Avg"] = np.nan
df["Sim_Mov_Avg"] = df["Adj Close"].rolling(window = span, min_periods= span).mean()

''' Exponential Moving Average '''
# df["Exp_MOv_Avg"] = np.nan
df["Exp_MOv_Avg"] = df["Adj Close"][span-1:].ewm(span = span, adjust = False).mean()

''' William %R '''
df["William%R"] = np.nan

''' Aroon Osc '''
df["Aroon Osc"] = np.nan

''' RSI '''
df["RSI"] = np.nan
df["RSI"] = rsiFunc(df["Adj Close"].values.tolist())


for i in range(len(df)):
    # if i != 0:
    #     df["return"].loc[i]= (df["Adj Close"].loc[i]/df["Adj Close"].loc[i-1])-1
    if i>=span-1:
        ''' William %R '''
        max_high = max(df["High"].loc[i-span:i])
        min_low = min(df["Low"].loc[i-span:i])
        close = df["Close"].loc[i]

        william_r = (max_high - close)/(max_high - min_low)
        df["William%R"].loc[i] = william_r*(-100)

        ''' Aroon '''
        max_index = df["High"].loc[i-span:i].values.tolist().index(max_high)
        min_index = df["Low"].loc[i-span:i].values.tolist().index(min_low)

        days_since_high = i - max_index
        days_since_low = i - min_index

        aroon_up = (span - days_since_high)*100/span
        aroon_dn = (span - days_since_low)*100/span

        df["Aroon Osc"].loc[i] = aroon_up - aroon_dn

pd.DataFrame.to_csv(df,filename+"_Features.csv",index=False)


# df= pd.read_csv("ICICInew.csv", parse_dates=["Date"])
# plt.plot( "Date","Mean", data=df,marker='o', color='r')
# plt.show()

# df.Date = pd.to_datetime(df.Date)

# print(df.info())
# print(df.describe())
# import pandas as pd
# rng = pd.date_range(start='2014-11-11', end='2014-11-26', freq='B')
# print(rng)

