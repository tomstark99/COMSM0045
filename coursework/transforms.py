import numpy as np
import torch

class FrequencyMasking(object):
    def __init__(self, p: float = 0.5, double = False):
        self.p = p
        self.double = double
    
    def __call__(self, sample):
        x = np.random.choice([0,1], p=[1-self.p, self.p])

        if x == 1:
            f = np.random.randint(0, 27)
            f0 = np.random.randint(0, len(sample[0]) - f)
            sample[:, f0:f0+f, :] = 0
            if self.double:
                f_ = np.random.randint(0, 27)
                f0_ = np.random.randint(0, len(sample[0]) - f_)
                sample[:, f0_:f0_+f_, :] = 0
    
        return sample

class TimeMasking(object):
    def __init__(self, p: float = 0.5, double = False):
        self.p = p 
        self.double = double
    
    def __call__(self, sample):
        x = np.random.choice([0,1], p=[1-self.p, self.p])

        if x == 1:
            t = np.random.randint(0, 101)
            t0 = np.random.randint(0, sample.size()[2] - t) 
            sample[:, :, t0:t0+t] = 0
            if self.double:
                t_ = np.random.randint(0, 101)
                t0_ = np.random.randint(0, sample.size()[2] - t_) 
                sample[:, :, t0_:t0_+t_] = 0

        return sample
