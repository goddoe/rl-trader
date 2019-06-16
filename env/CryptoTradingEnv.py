import random
import gym
from gym import spaces
import pandas as pd
import numpy as np
from stockstats import StockDataFrame
import mfsl

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 20000000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

MIN_BUY_PRICE = 1000
COMMISSION = 0.0005

INITIAL_ACCOUNT_BALANCE = 1000000


def naive_normalize(df):
    return (df-df.mean()) / (df.std() + 1e-6)


class CryptoTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, win_size=30):
        super(CryptoTradingEnv, self).__init__()

        self.df = StockDataFrame.retype(df)
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        self.win_size = win_size

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 3]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(90, ), dtype=np.float16)

        self.factory = mfsl.create_factory()
        self.df["lhc"] = self.df["close"]
        self.feats_str = feats_str = ["bollp(col(close))", "rsi(14)",
                                      "delta(ma(col(close), 26), ma(col(close), 12))"]
        features = [self.factory.parse_from(feat) for feat in feats_str]
        for feat_str, feat in zip(feats_str, features):
            self.df[feat_str] = feat(self.df)

    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1

        frag = self.df[self.current_step: self.current_step + self.win_size]

        market_feature = np.concatenate(list(map(naive_normalize,
                                                 (frag[f] for f in self.feats_str))), axis=0)
        market_feature = np.nan_to_num(market_feature) + 1e-6

        # state_feature = np.array([
        #     self.balance / MAX_ACCOUNT_BALANCE,
        #     # self.max_net_worth / MAX_ACCOUNT_BALANCE,
        #     self.shares_held / MAX_NUM_SHARES,
        #     # self.cost_basis / MAX_SHARE_PRICE,
        #     # self.total_shares_sold / MAX_NUM_SHARES,
        #     # self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        # ])

        # feature = np.concatenate([market_feature, state_feature], axis=0)

        return market_feature.flatten()

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = random.uniform(
            self.df.loc[self.current_step, "open"], self.df.loc[self.current_step, "close"])

        action_type = action[0]
        amount = action[1]

        if action_type < 1:
            # Buy amount % of balance in shares
            total_possible = self.balance / (current_price*(1. + COMMISSION))
            shares_bought = total_possible * amount
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * (current_price*(1.+COMMISSION))

            self.balance -= additional_cost
            self.cost_basis = (
                prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought

        elif action_type < 2:
            # Sell amount % of shares held
            shares_sold = self.shares_held * amount
            self.balance += shares_sold * (current_price*(1. - COMMISSION))
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        already_done = False
        if self.current_step > len(self.df.loc[:, 'open'].values) - self.win_size:
            self.current_step = 0
            already_done = True

        delay_modifier = (self.current_step / MAX_STEPS)

        reward = self.balance * delay_modifier

        done = self.net_worth <= 0

        if already_done:
            done = True

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self, init_step=None):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(
            0, len(self.df.loc[:, 'open'].values) - self.win_size)

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
