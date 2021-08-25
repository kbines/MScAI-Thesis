import numpy as np
import pandas as pd
import math
import random

from scipy import stats as scipy_stats
import gym
from gym import spaces
from gym.utils import seeding

from sklearn.preprocessing import StandardScaler

import stable_baselines3

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

class Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 date_from = '2008-01-01',
                 date_to = '2017-12-31',
                 filename='sp500.csv',
                 tensorboard_log="tensorboard",
                 investment=1000000,
                 risk_free_rate=0.5,
                 lookback=253,
                 sample_size=450,
                 random_sample=False,
                 report_point=np.iinfo(np.int32).max,  # default is don't report
                 reward_function='daily_returns'):
        self.filename = filename
        self.date_from = date_from
        self.date_to = date_to
        self.sample_size = sample_size
        self.random_sample = random_sample
        self.base_data = pd.read_csv(self.filename, sep=',', parse_dates=['date'],
                           usecols=['tic', 'date', 'open', 'low', 'high', 'close', 'adj_close',
                                    'daily_returns','ema_50', 'ema_200', 'bb_bbm', 'bb_bbh', 'bb_bbl','bb_bbhi', 'bb_bbli', 'stoch', 'stoch_signal', 'macd','macd_signal', 'obv', 'daily_returns'])
        self.base_data = self.base_data[(self.base_data.date >= self.date_from) & (self.base_data.date < self.date_to)]
        self.observation_attributes = \
            ['daily_returns','ema_50', 'ema_200', 'bb_bbm', 'bb_bbh', 'bb_bbl','bb_bbhi', 'bb_bbli', 'stoch', 'stoch_signal', 'macd','macd_signal', 'obv']
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

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.state_space, 1, len(self.observation_attributes)),dtype=np.float32)
        #self.observation_space = spaces.Box(low=-np.inf, high=np.inf,shape=(self.portfolio_asset_dim*len(self.observation_attributes),),dtype=np.float32)


    def data_split(self, data):
        if self.random_sample:
            # random sample of tic
            tics = np.random.choice(a=data.tic.unique(), size=self.sample_size, replace=False)
        else:
            # Sample assets by largest close price on the first day in the dataframe
            min_dt = data.date.min()
            min_df = data.loc[(data['date'] == min_dt)]
            tics = min_df.nlargest(self.sample_size, 'adj_close')['tic']

        data = data[data['tic'].isin(tics)]
        #fix nans
        data['daily_returns'].fillna(0, inplace=True)
        # normalize tech indicators
        #scaler = StandardScaler()
        #data.iloc[:,7:-1] = scaler.fit_transform(data.iloc[:,7:-1].to_numpy())
        # Index
        data.sort_values(['date', 'tic'], ignore_index=True, inplace=True )
        data.index = data.date.factorize()[0]

        return data

    def reset(self):

        self.terminal = False
        self.day = 0

        # resample data
        self.data = self.data_split(self.base_data)

        self.state = self.data.loc[self.day, :][self.observation_attributes].values[:, np.newaxis, :]#.to_numpy(dtype=np.float32).flatten()
        self.portfolio_value = self.investment

        self.sharpe = 0
        #self.sortino = 0
        self.psr = 0


        # init  history
        self.actions_history = np.zeros((self.trading_days, self.portfolio_asset_dim))
        self.weights_history = np.zeros((self.trading_days, self.portfolio_asset_dim))
        self.portfolio_value_history = np.zeros(self.trading_days)
        self.portfolio_value_history[0] = self.portfolio_value
        self.holdings_history = np.zeros((self.trading_days, self.portfolio_asset_dim))
        self.holdings_value_history = np.zeros((self.trading_days, self.portfolio_asset_dim))
        self.portfolio_returns_history = np.zeros(self.trading_days)
        self.cash_history = np.zeros(self.trading_days)
        #self.adj_returns_history = np.zeros(self.trading_days)
        self.cumulative_returns_history = np.zeros(self.trading_days)
        self.sharpe_history = np.zeros(self.trading_days)
        #self.sortino_history = np.zeros(self.trading_days)
        self.psr_history = np.zeros(self.trading_days)
        return self.state

    def step(self, actions):

        self.terminal = self.day >= self.trading_days - 1

        if self.terminal:

            self.render()

            self.info = {'actions_history': self.actions_history,
                         'weights_history': self.weights_history,
                         'holdings_history': self.holdings_history,
                         'holdings_value_history': self.holdings_value_history,
                         'cash_history' : self.cash_history,
                         'portfolio_value_history': self.portfolio_value_history,
                         'portfolio_returns_history': self.portfolio_returns_history,
                         #'adj_returns_history': self.adj_returns_history,
                         'cumulative_return_history': self.cumulative_returns_history,
                         'sharpe_history': self.sharpe_history,
                         #'sortino_history': self.sortino_history,
                         'psr_history': self.psr_history,
                         }
            # resample data
            self.data = self.data_split(self.base_data)

            return self.state, self.reward, self.terminal, self.info


        else:
            #set cash to 0
            self.cash = 0
            # Distrbute actions as weights summing to 1
            # Use softmax/relu so that 0 ation will still result in an non-zero allocation
            # We want to maintain a position in all selected assets
            # for small portfolios all action could be 0  so need to set action sum to 1 to avoid divide by 0
            action_sum = actions.sum()
            action_sum = action_sum if action_sum > 0 else 1
            #action_exp = np.exp(actions)
            #action_sum = np.sum(action_exp)
            #self.weights = np.true_divide(action_exp, action_sum )
            self.weights = np.true_divide(actions, action_sum)


            # Get asset prices for the trade day
            trade_day = self.data.loc[self.day, :]
            trade_closing_prices = trade_day.adj_close.to_numpy()

            # apply new weights to portfolio)
            #avoid divide by 0
            self.holdings = np.true_divide((self.weights * self.portfolio_value), trade_closing_prices, where=trade_closing_prices > 0, out=np.zeros_like(trade_closing_prices))

            #Re-Alloacte allocations in non-tradable assets (price = 0) to cash
            non_tradable_assets =  trade_closing_prices == 0
            self.cash = self.weights[non_tradable_assets].sum() * self.portfolio_value

            #self.holdings = (self.weights * self.portfolio_value) / trade_closing_prices

            # Move to next day (returns day)
            self.day += 1

            # Effective lookback
            self.effective_lookback = min(self.lookback, self.day)
            self.lookback_day = self.day - self.effective_lookback
            # Get returns dat data
            returns_day = self.data.loc[self.day, :]
            returns_closing_prices = returns_day.adj_close.to_numpy()
            self.holdings_value = self.holdings * returns_closing_prices

            # Portfolio value os the sum of the holding value plus cash
            self.portfolio_value = np.sum(self.holdings_value) + self.cash

            # Get returns proportional to the new weights
            weighted_returns = self.weights * returns_day.daily_returns

            # Portfolio returns are the sum of the weighted returns
            self.daily_portfolio_returns = weighted_returns.sum() * 100

            #Cumulative Returns - Gains since start of investment period
            self.cumulative_returns = (self.portfolio_value - self.investment) / self.investment * 100

            # Effective risk free rate is the risk free rate and risk free rate across portfolio age
            self.effective_rfr = self.day / self.lookback * self.risk_free_rate if self.day < self.lookback else self.risk_free_rate

            # Annualized return - sortino
            #self.annualized_return = (self.portfolio_value / self.investment) ** (self.effective_lookback / self.day)
            # Adjusted return - sortino
            #self.adj_returns = self.daily_portfolio_returns - self.effective_rfr
            #self.adj_returns_history[self.day] = self.adj_returns


            # save history
            self.actions_history[self.day] = actions
            self.weights_history[self.day] = self.weights
            self.holdings_history[self.day] = self.holdings
            self.holdings_value_history[self.day] = self.holdings_value
            self.portfolio_value_history[self.day] = self.portfolio_value
            self.portfolio_returns_history[self.day] = self.daily_portfolio_returns
            self.cumulative_returns_history[self.day] = self.cumulative_returns

            # Get portfolio performance indicators
            self.sharpe = self.get_sharpe()
            #self.sortino = self.get_sortino()
            #self.psr = self.get_psr()
            self.sharpe_history[self.day] = self.sharpe
            #self.sortino_history[self.day] = self.sortino
            #self.psr_history[self.day] = self.psr

            # Get new state
            self.state = trade_day[self.observation_attributes].values[:, np.newaxis, :]#.to_numpy(dtype=np.float32).flatten()
            # Reward functions
            self.reward_functions = {
                'daily_returns' : self.daily_portfolio_returns,
                'cum_returns': self.cumulative_returns,
                'portfolio_value': self.portfolio_value,
                'sharpe': self.sharpe
                #'sortino': self.sortino,
                #'psr': self.psr
            }

            self.reward = self.reward_functions.get(self.reward_function)

            if self.day % self.report_point == 0:
                self.render()

            # obs, reward, done, info
        return self.state, self.reward, self.terminal,{}

    def get_sharpe(self):
        # https://www.investopedia.com/terms/s/sharperatio.asp
        std = self.portfolio_returns_history[self.lookback_day:self.day].std()
        base_value = self.portfolio_value_history[self.lookback_day]
        current_returns = (self.portfolio_value - base_value) / base_value * 100 if base_value > 0 else 0
        sharpe = (current_returns - self.effective_rfr) / std if std > 0 else 0

        #if self.day % self.report_point == 0:
        #    print(f'day: {self.day}, Eff LB: {self.effective_lookback}, EEFR: {self.effective_rfr}, std: {std}, BD: {self.lookback_day}, BV: {base_value}, CR: {current_returns}, SR: {sharpe}')

        return sharpe
    """
    def get_sortino(self):
        # https://www.investopedia.com/terms/s/sortinoratio.asp

        # Downside risk
        # https://www.investopedia.com/terms/d/downside-deviation.asp
        # 1 -Get adjusted returns
        lookback_returns = self.adj_returns_history[self.lookback_day:self.day]

        #2 - Get negative returns
        neg_returns = lookback_returns[lookback_returns < 0]

        #3 - Downside Deviation - deviation of the negative returns across all returns ( hence don't use std)
        downside_deviation = np.sqrt(np.square(neg_returns).sum() / self.effective_lookback)

        if downside_deviation == 0:
            sortino = 0
        else:
            sortino = (self.annualized_return*100 - self.effective_rfr) / downside_deviation

        return sortino
    """

    def get_psr(self):
        # Probalistic Sharpe Ratio -
        # - https://quantdare.com/probabilistic-sharpe-ratio/
        # - https://github.com/rubenbriones/Probabilistic-Sharpe-Ratio/
        # - Marcos López de Prado and David Bailey (2012). The Sharpe ratio efficient frontier.
        # https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643
        # def probabilistic_sharpe_ratio(returns=None, sr_benchmark=0.0, *, sr=None, sr_std=None):

        # Get Std Deviation of sharpe ratio
        returns = pd.DataFrame(self.portfolio_returns_history[self.lookback_day:self.day])
        skew = pd.Series(scipy_stats.skew(returns), index=returns.columns)
        kurtosis = pd.Series(scipy_stats.kurtosis(returns, fisher=False), index=returns.columns)
        sr = self.sharpe
        sr_std = np.sqrt((1 + (0.5 * sr ** 2) - (skew * sr) + (((kurtosis - 3) / 4) * sr ** 2)) / (self.effective_lookback - 1))
        psr = sr_std.values[0] if not math.isinf(sr_std.values[0]) else 0

        return psr

    def render(self, mode='human', close=False):
        #sortino: {self.sortino:.3f}  \
        print(f'day: {self.day} \
                reward: {self.reward:.3f} \
                sharpe: {self.sharpe:.3f}  \
                psr: {self.psr:.3f}  \
                cum. rtns: {self.cumulative_returns:,.3f} \
                portf val: {self.portfolio_value:,.2f}')

        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        venv = DummyVecEnv([lambda: self])
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True)
        obs = venv.reset()
        venv = VecMonitor(venv)
        return venv, obs
