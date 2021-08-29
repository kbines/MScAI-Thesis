import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
from ta.momentum import StochasticOscillator
from ta.volume import OnBalanceVolumeIndicator
from ta.others import DailyReturnIndicator,CumulativeReturnIndicator


def set_tas(df):
# Add Tech Indicators
    # Exponential Moving Average
    ema_50 = EMAIndicator(close=df["adj_close"], window=50, fillna=True)
    ema_200 = EMAIndicator(close=df["adj_close"], window=200, fillna=True)
    df["ema_50"] = ema_50.ema_indicator()
    df["ema_200"] = ema_200.ema_indicator()

    # Bollinger
    bollinger = BollingerBands(close=df["adj_close"], window=20, window_dev=2, fillna=True)
    # Bollinger Bands
    df["bb_bbm"] = bollinger.bollinger_mavg()
    df["bb_bbh"] = bollinger.bollinger_hband()
    df["bb_bbl"] = bollinger.bollinger_lband()
    # Bollinger Band crossing
    df["bb_bbhi"] = bollinger.bollinger_hband_indicator()
    df["bb_bbli"] = bollinger.bollinger_lband_indicator()

    # Stochastic
    stochastic = StochasticOscillator(close=df["adj_close"], high=df["high"],
                                      low=df["low"], window=14, smooth_window=7, fillna=True)
    df["stoch"] = stochastic.stoch()
    df["stoch_signal"] = stochastic.stoch_signal()

    # MACD
    macd = MACD(close=df["adj_close"], window_fast=12, window_slow=26, window_sign=9, fillna=True)
    df["macd"] = macd.macd()
    df["macd_diff"] = macd.macd_diff()
    df["macd_signal"] = macd.macd_signal()

    # On Balance Volume
    obv = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"], fillna=True)
    df["obv"] = obv.on_balance_volume()

    # Returns
    daily_returns = DailyReturnIndicator(close=df["adj_close"])
    df["daily_returns"] = daily_returns.daily_return()

    # Market Cap
    df["market_cap"] = df["shares_os"]*df["adj_close"]

    return df

# get data
simfin = pd.read_csv('us-shareprices-daily.csv', sep=';', parse_dates=["Date"])

simfin = simfin.rename(columns = {'Ticker':'tic',
                         'SimFinId':'simfinid',
                         'Date':'date',
                         'Open':'open',
                         'Low':'low',
                         'High':'high',
                         'Close':'close',
                         'Adj. Close':'adj_close',
                         'Dividend':'dividend',
                         'Volume':'volume',
                         'Shares Outstanding':'shares_os'})


sp500_ticker_df=pd.read_csv('10025045_constituents requests.csv')
sp500_ticker_df=sp500_ticker_df.loc[sp500_ticker_df["Index Code"] == 500]
sp500_tickers = sp500_ticker_df.TICKER.unique().tolist()

sp500 = simfin[simfin['tic'].isin(sp500_tickers)]
sp500 = sp500.sort_values(by=['tic', 'date'],ignore_index=True)
# get unique dates
dates = sp500.date.unique()


count = 0
# Create Dataframe fr every date and ticker
progress = tqdm(total=(len(sp500_tickers)))
for tic in sp500_tickers:
    count += 1
    progress.update(n=1)
    # one row for each tic/date

    df = pd.DataFrame(product([tic],dates),columns=['tic','date'])
    df = pd.merge(df, sp500.loc[sp500.tic==tic], on=['tic','date'], how='left')
    for r in range(0, len(df)-1):
        if np.isnan(df['close'].iloc[r]):
            if r == 0:
                df['open'] = 0
                df['high'] = 0
                df['low'] = 0
                df['close'] = 0
                df['adj_close'] = 0
                df['shares_os'] = 0
            #fill in nulls - if same tic use previous values else use 0's
            elif df['tic'].iloc[r] == df['tic'].iloc[r-1]:
                df['open'].iloc[r] = df['open'].iloc[r-1]
                df['high'].iloc[r] = df['high'].iloc[r-1]
                df['low'].iloc[r] = df['low'].iloc[r-1]
                df['close'].iloc[r] = df['close'].iloc[r-1]
                df['adj_close'].iloc[r] = df['close'].iloc[r-1]
                df['shares_os'].iloc[r] = df['shares_os'].iloc[r - 1]

            #always set volume to 0
            df['volume'].iloc[r] = 0

    df = set_tas(df)
    final = pd.concat([final,df]) if count > 1 else df

final = final.sort_values(by=['tic', 'date'],ignore_index=True)

final.to_csv('sp500.csv')





