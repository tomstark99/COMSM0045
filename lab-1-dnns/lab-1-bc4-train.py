import torch as t
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from typing import Callable
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

class MLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_layer_dim: int,
        output_dim: int,
        activation_fn: Callable[[t.Tensor], t.Tensor] = F.relu
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_layer_dim)
        self.fc2 = nn.Linear(hidden_layer_dim, output_dim)
        self.activation_fn = activation_fn
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x
    
def accuracy(pred: t.FloatTensor, targets: t.LongTensor) -> float:
    if (pred.shape[0] != targets.shape[0]):
        raise ValueError(f"prediction tensor was of shape {pred.shape[0]}"f" but expected shape {targets.shape[0]}")
        
    correct = (pred.argmax(1) == targets).sum().item()
    return correct / pred.shape[0]

def main():

    iris = datasets.load_iris()
    p_processed = (iris['data']-iris['data'].mean())

    labels = iris['target']

    device = t.device('cuda')

    train_data, test_data, train_labels, test_labels = train_test_split(p_processed, labels, test_size=1/3)

    data = {
        'train': t.tensor(train_data, dtype=t.float32),
        'test': t.tensor(test_data, dtype=t.float32)
    }
    labels = {
        'train': t.tensor(train_labels, dtype=t.long),
        'test': t.tensor(test_labels, dtype=t.long)
    }

    features = 4
    hidden_layer_dim = 100
    class_count = 3

    model = MLP(features, hidden_layer_dim, class_count).to(device)
    optimiser = optim.SGD(model.parameters(), lr = 0.05)

    writer = SummaryWriter('logs', flush_secs=5)

    for epoch in range(0, 200):
        
        optimiser.zero_grad()    
        logits = model.forward(data['train'].to(device))

        loss = F.cross_entropy(logits, labels['train'].to(device))
        loss.backward()
        
        optimiser.step()
        
        train_accuracy = accuracy(logits, labels['train'].to(device)) * 100
        writer.add_scalar('accuracy/train', train_accuracy, epoch)
        writer.add_scalar('loss/train', loss.item(), epoch)

    writer.close()

if __name__ == "__main__":
    main()
