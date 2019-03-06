import gym
import gym_rlbot


class RaLLy():
    def __init__(self, name):
        self.name = name

    def train(self, env):
        while True:
            env.step(env.action_space.sample()) # take a random action

        # print(f"Agent name: {self.name}")
        # obs = env.reset()
        # print(obs)
        # for _ in range(1000):

if __name__ == "__main__":
    env = gym.make('RLBotEnv-v0')
    print(env.action_space)
    print(env.observation_space)
    agent = RaLLy("WaLLy")
    agent.train(env)
