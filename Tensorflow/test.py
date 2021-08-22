import gym
import PortfolioAllocationGym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
#%%
env_kwargs = {'filename':'sp500.csv',
    'date_from':'2008-01-01',
    'date_to':'2017-12-31',
    'investment':1000000,
    'risk_free_rate': 0.5, # approx US Treasury Note return
    'sample_size':500,
    'report_point':252, # 1 year
    'reward_function':'portfolio_value'}

train_env = gym.make('PortfolioAllocation-v0', **env_kwargs)
check_env(train_env)
venv, obs = train_env.get_sb_env()
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
policy_kwargs = {"n_steps": 5, "ent_coef": 0.005, "learning_rate": 0.0002}
a2c_model = A2C(policy=MlpPolicy,env=train_env, **policy_kwargs)
from PortfolioAllocationGym.callbacks import TensorBoardCallback as tbc
from datetime import datetime
# Random Agent, before training
# mean_reward, std_reward = evaluate_policy(a2c_model, train_env, n_eval_episodes=5)
# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
total_timesteps = 10 * (len(train_env.data.date.unique()) - 1)
trained_a2c_model = a2c_model.learn(total_timesteps=total_timesteps,tb_log_name='A2C' + datetime.now().strftime("%H-%M"))

