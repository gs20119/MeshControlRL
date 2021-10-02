
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
from utils import *
from noise import *
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
mem_maxlen = 50000
discount_factor = 0.99
actor_lr = 1e-4
critic_lr = 5e-4

tau = 1e-3

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(state_size, 128), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(128, action_size), nn.Tanh())

    def forward(self, state):
        x = self.layer1(state)
        x = self.layer2(x)
        return self.layer3(x)


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.layer1 = nn.Sequential(nn.Linear(state_size, 128), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(128+action_size, 128), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(128, 1))

    def forward(self, state, action):
        x = self.layer1(state)
        x = torch.cat([x, action], dim=-1)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class Agent:
    def __init__(self, state_size, action_size, train_mode_, load_model_):
        self.train_mode = train_mode_
        self.load_model = load_model_

        self.state_size = state_size
        self.action_size = action_size

        self.actor = Actor(self.state_size, self.action_size).to(device)
        self.critic = Critic(self.state_size, self.action_size).to(device)

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        self.OU = OU_noise(action_size)
        self.memory = deque(maxlen=mem_maxlen)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

    def get_action(self, state):
        action = self.actor(convertToTensorInput(state, self.state_size)).cpu().detach().numpy()
        noise = self.OU.sample()

        if self.train_mode: return action + noise
        else: return action

    def append_sample(self, state, action, rewards, next_state, done):
        self.memory.append((state, action, rewards, next_state, done))

    def train_model(self):
        if batch_size> len(self.memory) : return None
        mini_batch = random.sample(self.memory, batch_size)

        states = torch.FloatTensor([sample[0] for sample in mini_batch]).to(device)
        actions = torch.FloatTensor([sample[1] for sample in mini_batch]).to(device)
        rewards = torch.FloatTensor([sample[2] for sample in mini_batch]).to(device)
        next_states = torch.FloatTensor([sample[3] for sample in mini_batch]).to(device)
        dones = torch.FloatTensor([sample[4] for sample in mini_batch]).to(device)

        target_actor_actions = self.target_actor(next_states)
        target_critic_predict_qs = self.target_critic(next_states, target_actor_actions)

        target_qs = [reward + discount_factor * (1 - done) *
                                target_critic_predict_q for reward, target_critic_predict_q, done in
                                zip(rewards, target_critic_predict_qs, dones)]
        target_qs = torch.FloatTensor([target_qs]).to(device)

        q_val = self.critic(states, actions)

        critic_loss = torch.mean((q_val-target_qs)**2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
 
        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update_target(self.target_critic, self.critic)
        self.soft_update_target(self.target_actor, self.actor)

        return actor_loss, critic_loss

    def soft_update_target(self, target, orign):
        for target_param, orign_param in zip(target.parameters(), orign.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * orign_param.data)