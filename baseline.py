import gym
import json
import datetime as dt
from pathlib import Path

from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env.CryptoTradingEnv import CryptoTradingEnv

import pandas as pd

df = pd.read_csv('./data/upbit/upbit-btckrw-240m.csv', index_col=False)
df = df.sort_values('timestamp')

total = len(df)
train_ratio = 0.6
test_ratio = 1. - train_ratio
n_train = int(total * train_ratio)

train_df = df[:n_train]
test_df = df[n_train:]

test_df = test_df.reset_index()

# The algorithms require a vectorized environment to run
train_env = DummyVecEnv([lambda: CryptoTradingEnv(train_df)])
test_env = DummyVecEnv([lambda: CryptoTradingEnv(test_df)])

model = PPO2(MlpLnLstmPolicy,
             train_env,
             verbose=1,
             nminibatches=1,
             tensorboard_log=Path("./tensorboard").name)
model.learn(total_timesteps=20000)


obs = train_env.reset()
train_env.current_step = 0
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = train_env.step(action)
    train_env.render()


obs = test_env.reset()
test_env.current_step = 0
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = test_env.step(action)
    test_env.render()
