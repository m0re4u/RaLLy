import torch
from model import Actor, Critic
import utils
import numpy as np

device = 'cpu'

class DDPGTrainer(object):
    def __init__(self):
        self.actor = Actor().to(device)
        self.actor_target = Actor().to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr= 0.0001)

        self.critic = Critic().to(device)
        self.critic_target = Critic().to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)

        self.loss = torch.nn.MSELoss()

    def train(self, replay_buffer, iterations, batch_size=64, discount=0.99, tau=0.001):

        for it in range(iterations):
            # Sample replay buffer
            smp = replay_buffer.sample(batch_size)
            x, y, u, r, d = smp
            state = torch.FloatTensor(x).to(device)
            action = torch.stack(u)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Compute the target Q value
            ac = self.actor_target(next_state)
            ac = torch.cat(ac, dim=1)
            target_Q = self.critic_target(next_state, ac)
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = self.loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, torch.cat(self.actor(state), dim=1)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)