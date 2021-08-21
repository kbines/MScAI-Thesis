#%%

import gym
import PortfolioAllocationGym
import numpy as np
env_kwargs = {'filename':'sp500.csv',
    'date_from':'2008-01-01',
    'date_to':'2017-12-31',
#    'date_to':'2009-12-31',
    'investment':1000000,
    'risk_free_rate': 0.5, # approx US Treasury Note return
    'sample_size':500,
    'report_point':252, # 1 year
    'reward_function':'sharpe'}

train_env =  gym.make('PortfolioAllocation-v0', **env_kwargs)
#venv, _ = train_env.get_sb_env()
from stable_baselines3.common.env_checker import check_env
check_env(train_env)
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy,MultiInputPolicy
a2c_model = A2C(policy = MlpPolicy,
                use_sde = True,
                env = train_env,
                tensorboard_log = 'tensorboard',  verbose = 0,
                n_steps = 5, ent_coef = 0.05, learning_rate =0.0002
                )
from PortfolioAllocationGym.callbacks import TensorBoardCallback as tbc
from datetime import datetime

total_timesteps = 1 * (len(train_env.data.date.unique())-1)

trained_a2c_model= a2c_model.learn(total_timesteps=total_timesteps,
                                   tb_log_name='A2C'+datetime.now().strftime("%H-%M"))

