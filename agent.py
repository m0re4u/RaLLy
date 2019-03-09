import gym
import gym_rlbot

import torch
import numpy as np
from ddpg import DDPGTrainer
from utils import ReplayMemory

import traceback
import sys

class RaLLy():
    def __init__(self, name, env):
        self.name = name
        self.env = env
        self.eps = 0.005
        self.max_timesteps = 10000
        self.explore_noise = 0.1
        self.batch_size = 32
        self.discount = 0.99
        self.tau = 0.005
        self.max_episode_steps = 200
        self.memory = ReplayMemory(10000)

    def train(self):
        policy = DDPGTrainer()
        total_timesteps = 0
        episode_timesteps = 0
        episode_num = 0
        episode_done = True
        episode_reward = 0

        while total_timesteps < self.max_timesteps:
            if episode_done:
                if total_timesteps != 0:
                    print(f"Total steps: {total_timesteps:12} | Episodes: {episode_num:3} | Total reward: {episode_reward}")
                    # TODO: get training stats
                    policy.train(self.memory, episode_timesteps, self.batch_size, self.discount, self.tau)

                # Reset environment
                episode_done = False
                episode_num += 1
                episode_timesteps = 0
                episode_reward = 0
                obs = env.reset()


            control, jump, boost, handbrake = policy.actor(torch.tensor(obs))
            action = torch.cat([control, jump, boost, handbrake])

            if self.explore_noise != 0:
                noise = np.random.normal(0, self.explore_noise, size=5)
                noise = torch.clamp(torch.Tensor(noise), -1, 1)
                noise = torch.cat([noise, torch.zeros(3)])
                action = action + noise
                action = torch.clamp(action, -1, 1)

            # Perform action
            new_obs, reward, done, _ = env.step(action.detach())
            episode_done = True if episode_timesteps + 1 == self.max_episode_steps else done
            done_bool = float(done)
            episode_reward += reward

            # Store data in replay buffer
            self.memory.push((obs, new_obs, action, reward, done_bool))

            obs = new_obs
            episode_timesteps += 1
            total_timesteps += 1


if __name__ == "__main__":
    env = gym.make('RLBotEnv-v0')
    print(env.action_space)
    print(env.observation_space)
    agent = RaLLy("WaLLy", env)
    try:
        agent.train()
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        env.manager.kill_sub_processes()

