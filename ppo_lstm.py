# PPO-LSTM
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

from init import weight_init

# Hyperparameters
LEARNING_RATE = 0.00001
GAMMA = 0.98
LMBDA = 0.95
EPS_CLIP = 0.1
K_EPOCH = 5
N_EPI = 1000
N_BATCH = 20
T_HORIZON = 128

FEATURE_SIZE = 26


def save_model(model, save_dir, epoch_i, batch_i, desc=''):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir,
                             '{}{}_{}_ckpt.tar'.format(desc,
                                                       epoch_i,
                                                       batch_i))
    try:
        state_dict = model.module.state_dict()
    except AttributeError:
        state_dict = model.state_dict()
    torch.save(state_dict, save_path)
    return save_path


class PPO(nn.Module):
    def __init__(self, writer=None):
        super(PPO, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(FEATURE_SIZE, 64)
        self.ln1 = nn.LayerNorm(64)
        self.lstm = nn.LSTM(64, 32)
        self.ln2 = nn.LayerNorm(32)
        self.fc_pi = nn.Linear(32, 3)
        self.fc_pi_amount = nn.Linear(32, 3)
        self.fc_v = nn.Linear(32, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.writer = writer
        self.global_i = 0

    def pi(self, x, hidden):
        x = x.view(-1, FEATURE_SIZE)
        x = F.relu(self.ln1(self.fc1(x)))
        x = x.view(-1, 1, 64)
        feature, lstm_hidden = self.lstm(x, hidden)
        feature = self.ln2(feature)
        x = self.fc_pi(feature)
        action = F.softmax(x, dim=2)

        amount = self.fc_pi_amount(feature)
        amount = torch.sigmoid(amount)
        return action, amount, lstm_hidden

    def v(self, x, hidden):
        x = F.relu(self.ln1(self.fc1(x)))
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        (s_lst, a_lst, r_lst,
         s_prime_lst, prob_a_lst, amount_lst,
         hidden_lst, done_lst) = [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, amount, hidden, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            amount_lst.append([amount])
            hidden_lst.append(hidden)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        (s, a, r,
         s_prime, done_mask, prob_a,
         amount) = (torch.tensor(s_lst, dtype=torch.float).cuda(),
                    torch.tensor(a_lst).cuda(),
                    torch.tensor(r_lst).cuda(),
                    torch.tensor(s_prime_lst, dtype=torch.float).cuda(),
                    torch.tensor(done_lst, dtype=torch.float).cuda(),
                    torch.tensor(prob_a_lst).cuda(),
                    torch.tensor(amount_lst).cuda())
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a, amount, hidden_lst[0]

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a, amount, (h1, h2) = self.make_batch()
        first_hidden = (h1.detach().cuda(), h2.detach().cuda())

        for i in range(K_EPOCH):
            v_prime = self.v(s_prime, first_hidden).squeeze(1)
            td_target = r + GAMMA * v_prime * done_mask
            v_s = self.v(s, first_hidden).squeeze(1)
            delta = td_target - v_s
            delta = delta.detach().cpu().numpy()

            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = GAMMA * LMBDA * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).cuda()

            pi, amount_a, _ = self.pi(s, first_hidden)
            pi_a = pi.squeeze(1).gather(1, a)
            amount_a = amount_a.squeeze(1).gather(1, a)

            # a/b == log(exp(a)-exp(b))
            ratio1 = torch.exp(torch.log(pi_a) - torch.log(prob_a))
            ratio2 = torch.exp(torch.log(amount_a) - torch.log(amount))  # check
            ratio = ratio1 + ratio2

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-EPS_CLIP, 1+EPS_CLIP) * advantage
            loss = -torch.min(surr1, surr2) + \
                F.smooth_l1_loss(v_s, td_target.detach())

            if self.writer:
                writer.add_scalar('loss', loss.mean().item(), self.global_i)
                writer.add_scalar('delta', delta.mean().item(), self.global_i)
                writer.add_scalar('action_ratio', ratio1.mean().item(), self.global_i)
                writer.add_scalar('amount_ratio', ratio2.mean().item(), self.global_i)

            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.optimizer.step()
            self.global_i += 1


if __name__ == '__main__':

    from stable_baselines.common.vec_env import DummyVecEnv
    from env.CryptoTradingEnv import CryptoTradingEnv
    import pandas as pd

    # Load data
    df = pd.read_csv('./data/upbit/upbit-btckrw-240m.csv', index_col=False)
    df = df.sort_values('timestamp')

    total = len(df)
    train_ratio = 0.6
    n_train = int(total * train_ratio)

    train_df = df[:n_train]
    test_df = df[n_train:].reset_index()

    # Make Env
    train_env = DummyVecEnv([lambda: CryptoTradingEnv(train_df)])
    test_env = DummyVecEnv([lambda: CryptoTradingEnv(test_df)])

    writer = SummaryWriter("./tensorboard/20190616_1/mlp_lstm_ppo")

    # for i, row in train_df.iterrows():
    #     writer.add_scalars('train_price', {'open': row['open'],
    #                                  'close': row['close'],
    #                                  'low': row['low'],
    #                                  'high': row['high']}, i)

    # for i, row in test_df.iterrows():
    #     writer.add_scalars('test_price', {'open': row['open'],
    #                                  'close': row['close'],
    #                                  'low': row['low'],
    #                                  'high': row['high']}, i)

    model = PPO(writer)
    model = model.cuda()
    model.apply(weight_init)
    score = 0.0
    print_interval = 1

    for n_epi in range(N_EPI):
        hidden = (torch.zeros([1, 1, 32], dtype=torch.float).cuda(),
                  torch.zeros([1, 1, 32], dtype=torch.float).cuda())
        s = train_env.reset()
        done = False

        for i in range(N_BATCH):
            for t in range(T_HORIZON):
                s_tensor = torch.from_numpy(s).float().cuda()
                prob, amount, hidden = model.pi(s_tensor, hidden)
                prob = prob.view(-1)
                amount = amount.view(-1)
                m = Categorical(prob)
                action = m.sample().item()
                amount = amount.detach().cpu().numpy()[action]
                a = [(action, amount)]
                s_prime, r, done, info = train_env.step(a)

                model.put_data(
                    (s, action, r, s_prime, prob[action].item(), amount, hidden, done))
                s = s_prime

                score += r
                if done:
                    break

            model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(
                n_epi, (score/print_interval).item()))
            score = 0.0

    # train_env.close()

    model.eval()

    # validation
    obs = train_env.reset()
    train_env.current_step = 0
    hidden = (torch.zeros([1, 1, 32], dtype=torch.float).cuda(),
              torch.zeros([1, 1, 32], dtype=torch.float).cuda())

    while True:
        obs = torch.from_numpy(obs).cuda()
        prob, amount, hidden = model.pi(s_tensor, hidden)
        prob = prob.view(-1)
        amount = amount.view(-1)
        m = Categorical(prob)
        action = m.sample().item()
        amount = amount.detach().cpu().numpy()[action]
        a = [(action, amount)]
        obs, r, done, info = train_env.step(a)
        if done:
            break

        train_env.render()

    obs = test_env.reset()
    test_env.current_step = 0
    hidden = (torch.zeros([1, 1, 32], dtype=torch.float).cuda(),
              torch.zeros([1, 1, 32], dtype=torch.float).cuda())
    while True:
        obs = torch.from_numpy(obs).cuda()
        prob, amount, hidden = model.pi(s_tensor, hidden)
        prob = prob.view(-1)
        amount = amount.view(-1)
        m = Categorical(prob)
        action = m.sample().item()
        amount = amount.detach().cpu().numpy()[action]
        a = [(action, amount)]
        obs, r, done, info = test_env.step(a)
        if done:
            break

        test_env.render()


    torch.save(model, "./ckpts/mlp_lstm.torch")
    
    save_model("./ckpts/mlp_lstm.torch")
