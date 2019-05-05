import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorboardX import SummaryWriter
import time

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01  # learning rate
EPSILON = 0.95  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 10000
'''
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(),
                              int) else env.action_space.sample().shape  # to confirm the shape
'''
N_ACTIONS = 4 # retrieve from custom environment buy, hold, sell, close
N_STATES = 6 # ohlcv + position direction
ENV_A_SHAPE = 0
PATH = "net.pkl"



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()


        self.feature = nn.Sequential(
            nn.Linear(N_STATES, 100),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, N_ACTIONS)
        )

        self.value = nn.Sequential(
            nn.Linear(100, 500),
            nn.ReLU(),
            nn.Linear(500, 1)
        )


    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)

        # split out value and advantage to form q target
        # because we want to know which frames are valuable to act on instead of the entire as a consequence
        return value + advantage  - advantage.mean()


class torchDQN(object):



    def __init__(self, tensorboard=False):
        self.tensorboard=tensorboard
        if(self.tensorboard):
            self.writer = SummaryWriter('logs/ddqn-{}'.format(int(time.time())))

        # normal dqn sequential
        '''
        self.eval_net = torch.nn.Sequential(
            torch.nn.Linear(N_STATES, 50),  # 50 is number of dense layer out
            torch.nn.ReLU(),
            torch.nn.Linear(50, N_ACTIONS)
        )

        self.target_net = torch.nn.Sequential(
            torch.nn.Linear(N_STATES, 50),  # 50 is number of dense layer out
            torch.nn.ReLU(),
            torch.nn.Linear(50, N_ACTIONS)
        )
        '''

        # upgrade to dueling dqn almost the same just split our the target net
        self.eval_net, self.target_net = Net(), Net()

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)

        if (os.path.exists(PATH)):
            checkpoint = torch.load(PATH)
            self.eval_net.load_state_dict(checkpoint['eval'])
            self.target_net.load_state_dict(checkpoint['target'])
            self.optimizer.load_state_dict(checkpoint['opt'])
            #self.eval_net.train() # ensure that model is still in training mode
            self.eval_net.eval()

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # initialize memory *2 cause old and new state + 2 cause action and reward

        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:  # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index

        else:  # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self, globalStep):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES]) # old state  till 4th column
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int)) # then 4th till 4th +1
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2]) # then 4th +1 till 4th +2
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:]) # -5 the remaining from behind is new state

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if(self.tensorboard):
            self.writer.add_scalar('loss', loss.item(), global_step=globalStep)

    def save(self):
        torch.save({"eval": self.eval_net.state_dict(),
                    "target": self.target_net.state_dict(),
                    "opt" : self.optimizer.state_dict()
                   }, PATH)  # save entire net

