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
    ema_50 = EMAIndicator(close=df["close"], window=50, fillna=True)
    ema_200 = EMAIndicator(close=df["close"], window=200, fillna=True)
    df["ema_50"] = ema_50.ema_indicator()
    df["ema_200"] = ema_200.ema_indicator()

    # Bollinger
    bollinger = BollingerBands(close=df["close"], window=20, window_dev=2, fillna=True)
    # Bollinger Bands
    df["bb_bbm"] = bollinger.bollinger_mavg()
    df["bb_bbh"] = bollinger.bollinger_hband()
    df["bb_bbl"] = bollinger.bollinger_lband()
    # Bollinger Band crossing
    df["bb_bbhi"] = bollinger.bollinger_hband_indicator()
    df["bb_bbli"] = bollinger.bollinger_lband_indicator()

    # Stochastic
    stochastic = StochasticOscillator(close=df["close"], high=df["high"],
                                      low=df["low"], window=14, smooth_window=7, fillna=True)
    df["stoch"] = stochastic.stoch()
    df["stoch_signal"] = stochastic.stoch_signal()

    # MACD
    macd = MACD(close=df["close"], window_fast=12, window_slow=26, window_sign=9, fillna=True)
    df["macd"] = macd.macd()
    df["macd_diff"] = macd.macd_diff()
    df["macd_signal"] = macd.macd_signal()

    # On Balance Volume
    obv = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"], fillna=True)
    df["obv"] = obv.on_balance_volume()

    # Returns
    daily_returns = DailyReturnIndicator(close=df["close"])
    df["daily_returns"] = daily_returns.daily_return()

    return df

# gete data
yahoo = pd.read_csv('yahoo_daily_sp500_prices.csv', sep=',', parse_dates=["date"])

#exclude beacus of error in yahoo data
yahoo = yahoo[~yahoo.tic.isin(['TIE', 'HNZ', 'CBE', 'GR'])]
#get unique tics
tics = yahoo.tic.unique()

# get unique dates
dates = yahoo.date.unique()


count = 0
# Create Dataframe fr every date and ticker
progress = tqdm(total=(len(tics)))
for tic in tics:
    count += 1
    progress.update(n=1)
    # one row for each tic/date

    df = pd.DataFrame(product([tic],dates),columns=['tic','date'])
    df = pd.merge(df, yahoo.loc[yahoo.tic==tic], on=['tic','date'], how='left')
    for r in range(0, len(df)-1):
        if np.isnan(df['close'].iloc[r]):
            if r == 0:
                df['open'] = 0
                df['high'] = 0
                df['low'] = 0
                df['close'] = 0
            #fill in nulls - if same tic use previous values else use 0's
            elif df['tic'].iloc[r] == df['tic'].iloc[r-1]:
                df['open'].iloc[r] = df['open'].iloc[r-1]
                df['high'].iloc[r] = df['high'].iloc[r-1]
                df['low'].iloc[r] = df['low'].iloc[r-1]
                df['close'].iloc[r] = df['close'].iloc[r-1]

            #always set volume to 0
            df['volume'].iloc[r] = 0

    df = set_tas(df)
    final = pd.concat([final,df]) if count > 1 else df

final.to_csv('all_sp500_ta.csv')




