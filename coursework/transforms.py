import numpy as np
import torch

class HorizontalFlip(object):

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample):
        x = np.random.choice([0,1], p=[1-self.p, self.p])
        y = np.random.choice([0,1], p=[1-self.p, self.p])
        if x == 1:
            sample = torch.flip(sample, dims=[2])
        if y == 1:
            # noise = np.random.randn(len(sample))
            noise = torch.randn(sample.size())
            sample += 2 * noise
        
        return sample

