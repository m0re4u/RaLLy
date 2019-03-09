import random
class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        while len(self.memory) >= self.capacity:
            # Remove first element if we exceed the capacity
            self.memory.pop(0)
        self.memory.append(transition)


    def sample(self, batch_size):
        sample_list = random.sample(self.memory, k=batch_size)
        batch_x = [x[0] for x in sample_list]
        batch_y = [x[1] for x in sample_list]
        batch_action = [x[2] for x in sample_list]
        batch_reward = [x[3] for x in sample_list]
        batch_done = [x[4] for x in sample_list]
        return batch_x, batch_y, batch_action, batch_reward, batch_done

    def __len__(self):
        return len(self.memory)