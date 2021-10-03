
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
from utils import *
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, action_dim)

    def forward(self, state):
        a = F.relu(self.layer1(state))
        a = F.relu(self.layer2(a))
        return torch.tanh(self.layer3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)

        self.layer4 = nn.Linear(state_dim + action_dim, 256)
        self.layer5 = nn.Linear(256, 256)
        self.layer6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.layer1(sa))
        q1 = F.relu(self.layer2(q1))
        q1 = self.layer3(q1)

        q2 = F.relu(self.layer4(sa))
        q2 = F.relu(self.layer5(q2))
        q2 = self.layer6(q2)
        return q1, q2

    def Q(self, state, action):
        sa = torch.cat([state, action], 1)
        q = F.relu(self.layer1(sa))
        q = F.relu(self.layer2(q))
        q = self.layer3(q)
        return q


class TD3(object):
    def __init__(
        self, state_dim, action_dim, discount=0.99, 
        tau=0.005, noise=0.05, noise_clip=0.5, delay=2, bufferSize=50000
    ):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.discount = discount
        self.tau = tau
        self.noise = noise
        self.noise_clip = noise_clip
        self.delay = delay

        self.replay = deque(maxlen = bufferSize)
        self.iteration = 0


    def get_action(self, state):
        state = torch.FloatTensor(np.reshape(state,(1,-1))).to(device) 
        action = self.actor(state).cpu().data.numpy().flatten()
        action += np.random.randn(len(action)) * self.noise
        return action


    def append_sample(self, state, action, rewards, next_state, done):
        self.replay.append((state, action, rewards, next_state, done))


    def sample(self, batch_size):
        batch = random.sample(self.replay, batch_size)
        return(
            torch.FloatTensor([sample[0] for sample in batch]).to(device),
            torch.FloatTensor([sample[1] for sample in batch]).to(device),
            torch.FloatTensor([sample[2] for sample in batch]).to(device),
            torch.FloatTensor([sample[3] for sample in batch]).to(device),
            torch.FloatTensor([sample[4] for sample in batch]).to(device)
        )


    def train(self, batch_size=128):
        self.iteration += 1

        # Sample Replay Buffer
        state, action, reward, next_state, done = self.sample(batch_size)

        with torch.no_grad():
            noise = (
                torch.randn_like(action)*self.noise
            ).clamp(-self.noise_clip, self.noise_clip)
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-1,1)

        # Compute Target Q & current Q estimate
        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (1-done) * self.discount * target_Q
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss and Optimize
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss and Optimize with delay
        actor_loss = -self.critic.Q(state, self.actor(state)).mean()

        if self.iteration % self.delay == 0:    
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update_target(self.critic_target, self.critic)
            self.soft_update_target(self.actor_target, self.actor)

        return actor_loss, critic_loss


    def soft_update_target(self, target, origin):
        for target_param, param in zip(target.parameters(), origin.parameters()):
            target_param.data.copy_((1-self.tau) * target_param.data + self.tau * param.data)


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
	    self.critic.load_state_dict(torch.load(filename + "_critic"))
	    self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
	    self.critic_target = copy.deepcopy(self.critic)

	    self.actor.load_state_dict(torch.load(filename + "_actor"))
	    self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
	    self.actor_target = copy.deepcopy(self.actor)