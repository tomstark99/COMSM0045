import torch
import os
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import pandas as pd

from torch.utils.tensorboard import SummaryWriter

from dataset import DCASE
from Trainer import Trainer
from CNN import CNN

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    clip_length = 3
    batch_size = 1
    root_dir_train = os.getcwd() + '/ADL_DCASE_DATA/dev'
    root_dir_val = os.getcwd() + '/ADL_DCASE_DATA/dev'

    summary_writer = SummaryWriter("xddddd", flush_secs=1)

    train_dataset = DCASE(root_dir_train, clip_length)
    val_dataset = DCASE(root_dir_val, clip_length)

    model = CNN(clip_length, train_dataset.get_num_clips(), batch_size=1)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size = batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size = batch_size)
    optim = Adam(model.parameters(), lr=3e-4)

    trainer = Trainer(
        model, train_loader, val_loader, nn.CrossEntropyLoss(), optim, summary_writer, device
    )

    trainer.train(2, 1)

if __name__ == '__main__':
    main()
