from collections import deque
from dataclasses import dataclass
import random as rd

@dataclass(slots=True) # Denying dynamic attributes for memory efficiency
class Experience:
    state: any
    action: any
    reward: float
    next_state: any
    done: bool


class ReplayBuffer():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences from the buffer.
        """
        batch = rd.sample(self.buffer, batch_size) # This might be a bottleneck (O(n) complexity)
        return batch