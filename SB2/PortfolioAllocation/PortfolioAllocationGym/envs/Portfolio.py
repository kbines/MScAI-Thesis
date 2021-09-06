
import numpy as np
import pandas as pd
import math
import random
from datetime import datetime
import json
import warnings
import os

from scipy import stats as scipy_stats
import gym
from gym import spaces
from gym.utils import seeding
from sklearn.preprocessing import StandardScaler
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize


pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings(action='ignore',
                        category=DeprecationWarning,
                        module='stable_baselines')
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

class Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 date_from = '2008-01-01',
                 date_to = '2017-12-31',
                 filename='sp500.csv',
                 tensorboard_log="tensorboard",
                 investment=1000000,
                 observations = ['daily_returns', 'ema_50', 'ema_200'],
                 risk_free_rate=0.5,
                 lookback=253,
                 sample_size=10,
                 random_sample=False,
                 report_point=np.iinfo(np.int32).max,
                 save_info=False,
                 info_dir = 'info',# default is don't report
                 reward_function='daily_returns'):
        warnings.filterwarnings(action='ignore',
                                category=DeprecationWarning,
                                module='stable_baselines')
        warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
        warnings.filterwarnings("ignore", category=UserWarning, module='gym')
        self.filename = filename
        self.date_from = date_from
        self.date_to = date_to
        self.sample_size = sample_size
        self.random_sample = random_sample
        self.observation_attributes = observations
        self.info_dir = info_dir



        if save_info:
            self.save_info = True
        else:
            self.save_info = False

        #get stock data
        #possible observation metrics are
        #['daily_returns', 'ema_50', 'ema_200', 'bb_bbm', 'bb_bbh', 'bb_bbl','bb_bbhi', 'bb_bbli', 'stoch', 'stoch_signal', 'macd','macd_signal', 'obv']
        usecols = ['tic', 'date', 'adj_close'] + self.observation_attributes
        self.base_data = pd.read_csv(self.filename, sep=',', parse_dates=['date'],usecols=usecols)
        self.base_data = self.base_data[(self.base_data.date >= self.date_from) & (self.base_data.date < self.date_to)]

        #Get index returns
        self.index_returns = pd.read_csv('sp500_returns.csv', sep=',', parse_dates=['Date'])
        self.index_returns = self.index_returns[(self.index_returns.Date >= self.date_from) & (self.index_returns.Date < self.date_to)]
        self.index_returns.set_index('Date', inplace=True)

        self.tensorboard_log = tensorboard_log
        self.investment = investment
        self.risk_free_rate = risk_free_rate

        self.lookback = lookback
        self.report_point = report_point
        self.reward_function = reward_function
        self.trading_days = len(self.base_data.date.unique()) - 1
        self.portfolio_asset_dim = sample_size

        self.state_space = self.portfolio_asset_dim

        # Action space
        self.action_space = spaces.Box(low=0, high=1, shape=(self.portfolio_asset_dim,))

        # Observation space - 1d flattened array of observations * assets
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,shape=(self.portfolio_asset_dim*len(self.observation_attributes),),dtype=np.float32)


    def data_split(self, data):
        if self.random_sample:
            # random sample of tic
            self.tics = np.random.choice(a=data.tic.unique(), size=self.sample_size, replace=False)
        else:
            # Sample assets by largest close price on the first day in the dataframe
            min_dt = data.date.min()
            min_df = data.loc[(data['date'] == min_dt)]
            self.tics = min_df.nlargest(self.sample_size, 'adj_close')['tic']

        data = data[data['tic'].isin(self.tics)]
        #fix nans
        data['daily_returns'].fillna(0, inplace=True)
        # Index
        data.sort_values(['date', 'tic'], ignore_index=True, inplace=True )
        data.index = data.date.factorize()[0]

        return data

    def reset(self):

        self.terminal = False
        self.day = 0

        # resample data
        self.data = self.data_split(self.base_data)

        self.state = self.data.loc[self.day, :][self.observation_attributes].to_numpy(dtype=np.float32).flatten()
        self.portfolio_value = self.investment

        self.sharpe = 0


        # init  history
        self.actions_history = np.zeros((self.trading_days, self.portfolio_asset_dim))
        self.weights_history = np.zeros((self.trading_days, self.portfolio_asset_dim))
        self.portfolio_value_history = np.zeros(self.trading_days)
        self.portfolio_value_history[0] = self.portfolio_value
        self.holdings_history = np.zeros((self.trading_days, self.portfolio_asset_dim))
        self.holdings_value_history = np.zeros((self.trading_days, self.portfolio_asset_dim))
        self.portfolio_returns_history = np.zeros(self.trading_days)
        self.cash_history = np.zeros(self.trading_days)
        self.cumulative_returns_history = np.zeros(self.trading_days)
        self.sharpe_history = np.zeros(self.trading_days)
        self.benchmark_history = np.zeros(self.trading_days)
        self.reward_history = np.zeros(self.trading_days)

        return self.state

    def step(self, actions):

        self.terminal = self.day >= self.trading_days - 1

        if self.terminal:
            self.render()

            self.info = {
                         'tics': self.tics,
                         'actions_history': self.actions_history,
                         'weights_history': self.weights_history,
                         'holdings_history': self.holdings_history,
                         'holdings_value_history': self.holdings_value_history,
                         'cash_history' : self.cash_history,
                         'portfolio_value_history': self.portfolio_value_history,
                         'portfolio_returns_history': self.portfolio_returns_history,
                         'benchmark_history': self.benchmark_history,
                         'cumulative_return_history': self.cumulative_returns_history,
                         'sharpe_history': self.sharpe_history,
                         'reward_history': self.reward_history
                         }
            # resample data if using random samples
            if self.random_sample:
                self.data = self.data_split(self.base_data)

            if self.save_info:
                self.info_json = {
                                'tics' : self.tics.tolist(),
                                'actions_history': self.actions_history.tolist(),
                                'weights_history': self.weights_history.tolist(),
                                'holdings_history': self.holdings_history.tolist(),
                                'holdings_value_history': self.holdings_value_history.tolist(),
                                'cash_history': self.cash_history.tolist(),
                                'portfolio_value_history': self.portfolio_value_history.tolist(),
                                'portfolio_returns_history': self.portfolio_returns_history.tolist(),
                                'benchmark_history': self.benchmark_history.tolist(),
                                'cumulative_return_history': self.cumulative_returns_history.tolist(),
                                'sharpe_history': self.sharpe_history.tolist(),
                                'reward_history': self.reward_history.tolist(),
                             }
                file_name = os.path.join(self.info_dir,self.reward_function+datetime.now().strftime("%m%d-%H%M%S")+'.json')
                with open(file_name,'w') as file:
                    json.dump(self.info_json,file,sort_keys=True, indent=4)


            return self.state, self.reward, self.terminal, self.info


        else:
            #set cash to 0
            self.cash = 0
            # Distrbute actions as weights summing to 1
            # Use softmax/relu so that 0 ation will still result in an non-zero allocation
            # We want to maintain a position in all selected assets
            action_sum = actions.sum()

            action_sum = action_sum if action_sum > 0 else 1
            self.weights = np.true_divide(actions, action_sum)

            # Get asset prices for the trade day
            trade_day = self.data.loc[self.day, :]
            trade_closing_prices = trade_day.adj_close.to_numpy()

            # apply new weights to portfolio)
            #avoid divide by 0
            self.holdings = np.true_divide((self.weights * self.portfolio_value), trade_closing_prices, where=trade_closing_prices > 0, out=np.zeros_like(trade_closing_prices))

            #Re-Alloacte allocations in non-tradable assets (price = 0) to cash
            non_tradable_assets =  trade_closing_prices == 0
            #if all actions are 0 then move all investments to cash
            self.cash = self.weights[non_tradable_assets].sum() * self.portfolio_value if actions.sum()> 0 else self.portfolio_value

            # Move to next day (returns day)
            self.day += 1

            # Effective lookback
            self.effective_lookback = min(self.lookback, self.day)
            self.lookback_day = self.day - self.effective_lookback
            # Get returns day data
            returns_day = self.data.loc[self.day, :]
            returns_closing_prices = returns_day.adj_close.to_numpy()
            self.holdings_value = self.holdings * returns_closing_prices

            # Portfolio value os the sum of the holding value plus cash
            self.portfolio_value = np.sum(self.holdings_value) + self.cash

            # Get returns proportional to the new weights
            weighted_returns = self.weights * returns_day.daily_returns

            # Portfolio returns are the sum of the weighted returns
            self.daily_portfolio_returns = weighted_returns.sum() * 100

            self.index_return =  self.index_returns.iloc[self.day].daily_returns * 100
            self.benchmark = self.daily_portfolio_returns - self.index_return
            #Cumulative Returns - Gains since start of investment period
            self.cumulative_returns = (self.portfolio_value - self.investment) / self.investment * 100

            # Effective risk free rate is the risk free rate and risk free rate across portfolio age
            self.effective_rfr = self.day / self.lookback * self.risk_free_rate if self.day < self.lookback else self.risk_free_rate

            # save history
            self.actions_history[self.day] = actions
            self.weights_history[self.day] = self.weights
            self.holdings_history[self.day] = self.holdings
            self.holdings_value_history[self.day] = self.holdings_value
            self.portfolio_value_history[self.day] = self.portfolio_value
            self.portfolio_returns_history[self.day] = self.daily_portfolio_returns
            self.benchmark_history[self.day] = self.benchmark
            self.cumulative_returns_history[self.day] = self.cumulative_returns

            # Get portfolio performance indicators
            self.sharpe = self.get_sharpe()
            self.sharpe_history[self.day] = self.sharpe


            # Get new state
            self.state = trade_day[self.observation_attributes].to_numpy(dtype=np.float32).flatten()#.values[:, np.newaxis, :]#.to_numpy(dtype=np.float32).flatten()
            # Reward functions
            self.reward_functions = {
                'daily_returns' : self.daily_portfolio_returns,
                'benchmark' : self.benchmark,
                'cum_returns': self.cumulative_returns,
                'portfolio_value': self.portfolio_value,
                'sharpe': self.sharpe
            }

            self.reward = self.reward_functions.get(self.reward_function)
            self.reward_history[self.day] = self.reward

            if self.day % self.report_point == 0:
                self.render()

        return self.state, self.reward, self.terminal,{}

    def get_sharpe(self):
        # https://www.investopedia.com/terms/s/sharperatio.asp
        std = self.portfolio_returns_history[self.lookback_day:self.day].std()
        base_value = self.portfolio_value_history[self.lookback_day]
        #current_returns = (self.portfolio_value - base_value) / base_value * 100 if base_value > 0 else 0
        #sharpe = (current_returns - self.effective_rfr) / std if std > 0 else 0
        sharpe = (self.portfolio_returns_history.mean() - self.effective_rfr) / std if std > 0 else 0
        return sharpe * 100

    def render(self, mode='human', close=False):
        #sortino: {self.sortino:.3f}  \
        print(f'day: {self.day} \
                reward: {self.reward_history.mean():.3f} \
                daily_returns: {self.portfolio_returns_history.mean():.3f} \
                benchmark: {self.benchmark_history.mean():.3f} \
                index : {self.index_return:.3f} \
                sharpe: {self.sharpe:.3f} \
                cum. rtns: {self.cumulative_returns:.3f} \
                portf val: {self.portfolio_value:,.2f}')

        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        venv = DummyVecEnv([lambda: self])
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True)
        obs = venv.reset()
        return venv, obs
