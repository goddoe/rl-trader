# actor critic
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

from init import weight_init

# Hyperparameters
LEARNING_RATE = 0.001
GAMMA = 0.98

FEATURE_SIZE = 96


class ActorCritic(nn.Module):
    def __init__(self, writer=None):
        super(ActorCritic, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(FEATURE_SIZE, 64)
        self.ln1 = nn.LayerNorm(64)
        self.fc_pi = nn.Linear(64, 3)
        self.fc_pi_amount = nn.Linear(64, 3)
        self.fc_v = nn.Linear(64, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.writer = writer
        self.global_i = 0

    def pi(self, x):
        x = x.view(-1, FEATURE_SIZE)
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.fc_pi(x)
        action = F.softmax(x, dim=1)
        return action

    def v(self, x):
        x = x.view(-1, FEATURE_SIZE)
        x = F.relu(self.ln1(self.fc1(x)))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        (s_lst, a_lst, r_lst,
         s_prime_lst, prob_a_lst,
         done_lst) = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        (s, a, r,
         s_prime, done_mask,
         prob_a) = (torch.tensor(s_lst, dtype=torch.float).cuda(),
                    torch.tensor(a_lst).cuda(),
                    torch.tensor(r_lst).cuda(),
                    torch.tensor(s_prime_lst, dtype=torch.float).cuda(),
                    torch.tensor(done_lst, dtype=torch.float).cuda(),
                    torch.tensor(prob_a_lst).cuda())
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()
        td_target = r + GAMMA * self.v(s_prime) * done_mask
        delta = td_target - self.v(s)

        pi = self.pi(s)
        pi_a = pi.gather(1, a)

        policy_loss = -torch.log(pi_a) * delta.detach()
        td_error = F.smooth_l1_loss(self.v(s), td_target.detach())

        loss = policy_loss + td_error

        if self.writer:
            writer.add_scalar('loss', loss.mean().item(), self.global_i)
            writer.add_scalar('td_error', td_error.mean().item(), self.global_i)
            writer.add_scalar('policy_loss', policy_loss.mean().item(), self.global_i)

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        self.global_i += 1


if __name__ == '__main__':

    from stable_baselines.common.vec_env import DummyVecEnv
    from env.CryptoTradingEnvAllIn2 import CryptoTradingEnvAllIn2
    from preparer import TickerDataFramePreparer
    import pandas as pd

    # Load data
    df = pd.read_csv('./data/upbit-btckrw-1m.csv', index_col=['timestamp'], parse_dates=['timestamp'])
    df = df.sort_values('timestamp')

    df = TickerDataFramePreparer(
        window='15m',
    ).prepare(df)

    total = len(df)
    train_ratio = 0.6
    n_train = int(total * train_ratio)

    train_df = df.iloc[:n_train].reset_index()
    test_df = df.iloc[n_train:].reset_index()

    # Make Env
    train_env = DummyVecEnv([lambda: CryptoTradingEnvAllIn2(train_df)])
    test_env = DummyVecEnv([lambda: CryptoTradingEnvAllIn2(test_df)])

    # writer
    writer = SummaryWriter("./tensorboard/20190617/actor_critic")

    model = ActorCritic(writer)
    model = model.cuda()
    model.apply(weight_init)
    print_interval = 1

    n_epi = 1000
    n_batch = 30
    n_rollout = 24

    for n_epi in range(n_epi):
        s = train_env.reset()
        done = False

        score = 0.0

        for i in range(n_batch):
            for t in range(n_rollout):
                s_tensor = torch.from_numpy(s).float().cuda()
                prob = model.pi(s_tensor)
                prob = prob.view(-1)
                m = Categorical(prob)
                action = m.sample().item()
                a = [action]
                s_prime, r, done, info = train_env.step(a)
                s, s_prime, r, done = s[0], s_prime[0], r[0], done[0] 

                model.put_data(
                    (s, action, r, s_prime, prob[action].item(), done))
                s = np.array([s_prime])

                score += r
                if done:
                    break

            model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(
                n_epi, (score/print_interval).item()))
            score = 0.0

    model.eval()

    # validation
    train_env.reset()
    train_env.current_step = 0
    obs, _, _, _ = train_env.step([2])

    prob_list = []

    while True:
        obs = torch.from_numpy(obs).cuda()
        prob = model.pi(s_tensor)
        prob = prob.view(-1)
        # m = Categorical(prob)
        # action = m.sample().item()
        prob_list.append(prob.detach().cpu().tolist())
        action = prob.argmax().item()

        a = [action]
        obs, r, done, info = train_env.step(a)

        if done:
            break

        train_env.render()

    test_env.reset()
    test_env.current_step = 0
    obs, _, _, _ = train_env.step([2])

    while True:
        obs = torch.from_numpy(obs).cuda()
        prob = model.pi(s_tensor)
        prob = prob.view(-1)
        m = Categorical(prob)
        action = m.sample().item()
        a = [action]
        obs, r, done, info = test_env.step(a)
        if done:
            break

        test_env.render()

    torch.save(model, "./ckpts/test.torch")
