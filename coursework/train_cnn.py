import torch
import os
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import argparse
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from dataset import DCASE
from trainer import Trainer
from CNN import CNN

parser = argparse.ArgumentParser(
    description="Train a simple CNN on CIFAR-10",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

default_dataset_dir = Path(os.getcwd()) / 'ADL_DCASE_DATA'

parser.add_argument(
    "--dataset-root", 
    default=default_dataset_dir, 
    type=Path
)
parser.add_argument(
    "--log-dir", 
    default=Path("logs"), 
    type=Path
)
parser.add_argument(
    "--learning-rate", 
    default=1e-1, 
    type=float, 
    help="Learning rate"
)
parser.add_argument(
    "--batch-size",
    default=64,
    type=int,
    help="Number of images within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=10,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--val-frequency",
    default=1,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=5,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=100,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)

def main(args):

    if torch.cuda.is_available():
        # set to "cpu" if testing on lab machine
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    clip_length = 3
    root_dir_train = args.dataset_root / 'development'
    root_dir_val = args.dataset_root / 'evaluation'

    summary_writer = SummaryWriter(args.log_dir, flush_secs=1)

    train_dataset = DCASE(root_dir_train, clip_length)
    val_dataset = DCASE(root_dir_val, clip_length)

    model = CNN(clip_length, train_dataset.get_num_clips())
    optim = Adam(model.parameters(), lr=args.learning_rate)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size = args.batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size = args.batch_size)

    trainer = Trainer(
        model, 
        train_loader, 
        val_loader, 
        nn.CrossEntropyLoss(), 
        optim, 
        summary_writer, 
        device
    )

    trainer.train(
        epochs=args.epochs, 
        val_frequency=args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency
    )

if __name__ == '__main__':
    main(parser.parse_args())
