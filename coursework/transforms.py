import numpy as np
import torch

class HorizontalFlip(object):

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample):
        x = np.random.choice([0,1], p=[1-self.p, self.p])
        if x == 1:
            return torch.flip(sample, dims=[2])
        else:
            return sample

