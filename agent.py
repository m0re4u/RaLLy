import gym
import gym_rlbot

from model import DQN
import torch

class RaLLy():
    def __init__(self, name, env):
        self.name = name
        self.env = env
        
        self.model = DQN()

    def train(self):
        obs = env.reset()
        while True:
            with torch.no_grad():
                control, jump, boost, handbrake = self.model(torch.tensor(obs))
            obs, reward, scored, _ = env.step([*control, jump.item(), boost.item(), handbrake.item()])


if __name__ == "__main__":
    env = gym.make('RLBotEnv-v0')
    print(env.action_space)
    print(env.observation_space)
    agent = RaLLy("WaLLy", env)
    agent.train()
