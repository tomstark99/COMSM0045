import numpy as np
import torch

class HorizontalFlip(object):

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample):
        x = np.random.choice([0,1], p=[1-self.p, self.p])
        #flip
        if x == 1:
            sample = torch.flip(sample, dims=[2])

        return sample

class RandomNoise(object):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample):
        y = np.random.choice([0,1], p=[1-self.p, self.p])
        # noise
        if y == 1:
            # noise = np.random.randn(len(sample))
            noise = torch.randn(sample.size())
            sample += 2 * noise

        return sample

class RandomSplit(object):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample):
        z = np.random.choice([0,1], p=[1-self.p, self.p])
        # split
        if z == 1:
            length = sample.shape[2]
            range_ = np.random.choice(np.arange(0, length))
            new, new2 = sample[:, :, 0:range_], sample[:, :, range_:length]
            sample = torch.cat([new2, new], dim=2)

        return sample

class FrequencyMasking(object):
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, sample):
        x = np.random.choice([0,1], p=[1-self.p, self.p])

        if x == 1:
            for i, s in enumerate(sample):
                f = np.random.randint(0, 27)
                f0 = np.random.randint(0, len(s) - f)

                sample[i, f0:f0+f, :] = s.mean()
    
        return sample

class TimeMasking(object):
    def __init__(self, p: float = 0.5):
        self.p = p 
    
    def __call__(self, sample):
        x = np.random.choice([0,1], p=[1-self.p, self.p])

        if x == 1:
            for i, s in enumerate(sample):
                t = np.random.randint(0, 20)
                t0 = np.random.randint(0, len(s[0]) - t)

                sample[i, :, t0:t0+t] = s.mean()
        
        return sample