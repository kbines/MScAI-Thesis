import pandas as pd
import yfinance as yf

from ta.others import DailyReturnIndicator,CumulativeReturnIndicator

# get data
gspc= yf.Ticker("^GSPC")
df = gspc.history(start= '2007-01-01', end= '2020-12-31')
daily_returns = DailyReturnIndicator(close=df["close"])
df["daily_returns"] = daily_returns.daily_return()

df.to_csv('sp500_returns.csv')




