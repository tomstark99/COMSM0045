from multiprocessing import cpu_count
from typing import Tuple
import argparse
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, SubsetRandomSampler
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
    default=1e-3, 
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
    default=50,
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
    default=10,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--num-workers",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)
parser.add_argument(
    "--use-cuda",
    default=True,
    type=bool,
    help="Use the GPU for training."
)
parser.add_argument(
    "--full-train",
    default=True,
    type=bool,
    help="Train on the full train/test dataset."
)
parser.add_argument(
    "--train-split",
    default=0.25,
    type=float,
    help="Percentage of the training data to split as a validation set."
)

def no_collate(args):
    return args

def train_test_loader(dataset: DCASE, batch_size: int, val_split: float) -> Tuple[DataLoader, DataLoader]:
    if val_split < 0.0 or val_split > 1.0:
        raise ValueError(f'training split should be between 0 and 1, but was {val_split}')

    idxs = list(range(len(dataset)))
    split = int(np.floor(val_split * len(dataset)))
    np.random.shuffle(idxs)

    train_idx, test_idx = idxs[split:], idxs[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    return DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=no_collate), DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, collate_fn=no_collate)

def main(args):

    if args.use_cuda and torch.cuda.is_available():
        # set to "cpu" if testing on lab machine
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    torch.cuda.empty_cache()

    clip_length = 3
    root_dir_train = args.dataset_root / 'development'
    root_dir_val = args.dataset_root / 'evaluation'

    log_dir = get_summary_writer_log_dir(args)
    print(f'writing logs to {log_dir}')

    summary_writer = SummaryWriter(str(log_dir), flush_secs=1)

    train_dataset = DCASE(root_dir_train, clip_length)
    val_dataset = DCASE(root_dir_val, clip_length)

    model = CNN(clip_length, train_dataset.get_num_clips())
    optim = Adam(model.parameters(), lr=args.learning_rate)

    if args.full_train:
        print("fksake")
        train_loader = DataLoader(
            train_dataset, 
            shuffle=True, 
            batch_size=args.batch_size,
            pin_memory=True,
            num_workers=args.num_workers
        )
        val_loader = DataLoader(
            val_dataset, 
            shuffle=False, 
            batch_size=args.batch_size,
            pin_memory=True,
            num_workers=args.num_workers
        )
    else:
        print("alie")
        train_loader, val_loader = train_test_loader(train_dataset, args.batch_size, args.train_split)

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
    trainer.print_per_class_accuracy()
    trainer.save_model_params()

def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """
    Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.
    Args:
        args: CLI Arguments
    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = f'CNN_bs={args.batch_size}_lr={args.learning_rate}_run_'

    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

if __name__ == '__main__':
    main(parser.parse_args())
