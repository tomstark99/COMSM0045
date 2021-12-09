from multiprocessing import cpu_count
from typing import Tuple
import argparse
from pathlib import Path
import os
import gc

import numpy as np
from numpy import testing
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset, random_split
from torch.utils.tensorboard import SummaryWriter

from dataset import DCASE, NF_DCASE, V_DCASE
from trainer import Trainer
from CNN import CNN
from torchvision.transforms import Compose
from transforms import FrequencyMasking, TimeMasking

parser = argparse.ArgumentParser(
    description="CW CNN training for DCASE 2016",
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
    default=False,
    action='store_true',
    help="Use the GPU for training."
)
parser.add_argument(
    "--full-train",
    default=False,
    action='store_true',
    help="Train on the full train/test dataset."
)
parser.add_argument(
    "--train-split",
    default=0.25,
    type=float,
    help="Percentage of the training data to split as a validation set."
)

parser.add_argument(
    "--freq-mask", 
    default=-1,
    type=float
)

parser.add_argument(
    "--time-mask", 
    default=-1,
    type=float
)

parser.add_argument(
    "--double",
    default=False,
    action="store_true"
)

parser.add_argument(
    "--normalise",
    default=False,
    action="store_true"
)

def train_test_loader(dataset: DCASE, batch_size: int, val_split: float, transforms) -> Tuple[DataLoader, DataLoader]:
    
    labels = {label: [clips for clips in dataset._labels[dataset._labels['label']==label].clip_no.unique()] for label in dataset._labels['label']}

    train = []
    test = []

    for label, clips in labels.items():
        total = list(range(len(clips)))
        np.random.shuffle(total)
        train_idx, test_idx = total[3:], total[:3]
        
        for i in train_idx:
            train.append(clips[i])
        for j in test_idx:
            test.append(clips[j])

    train_clips = []
    test_clips = []

    for tr in train:
        temp = dataset._labels[dataset._labels['clip_no'] == tr].file
        train_clips.extend(temp)
        
    for te in test:
        temp = dataset._labels[dataset._labels['clip_no'] == te].file
        test_clips.extend(temp)

    train_subset = NF_DCASE(
        dataset._root_dir, 
        dataset._clip_duration, 
        train_clips,
        transforms,
        args.normalise
    )
    val_subset = NF_DCASE(
        dataset._root_dir, 
        dataset._clip_duration, 
        test_clips,
        args.normalise
    )

    return DataLoader(train_subset, batch_size=batch_size, shuffle=True), DataLoader(val_subset, batch_size=batch_size, shuffle=False)

def main(args):
    
    gc.collect()
    torch.cuda.empty_cache()

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.cuda.empty_cache()

    clip_length = 3
    root_dir_train = args.dataset_root / 'development'
    root_dir_val = args.dataset_root / 'evaluation'

    log_dir = get_summary_writer_log_dir(args)
    print(f'writing logs to {log_dir}')

    summary_writer = SummaryWriter(str(log_dir), flush_secs=1)
    
    transform = None
    transforms_ = []
    if args.freq_mask != -1:
        transforms_.append(FrequencyMasking(args.freq_mask, args.double))
    if args.time_mask != -1:
        transforms_.append(TimeMasking(args.freq_mask, args.double))
    
    if transforms_:
        transform = Compose(transforms_)

    train_dataset = DCASE(root_dir_train, clip_length, transform=transform, normalise = args.normalise)

    model = CNN(clip_length, train_dataset.get_num_clips())
    optim = Adam(model.parameters(), lr=args.learning_rate)

    if args.full_train:
        val_dataset = V_DCASE(root_dir_val, clip_length, args.normalise)

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
        train_loader, val_loader = train_test_loader(train_dataset, args.batch_size, args.train_split, transform)

    trainer = Trainer(
        model, 
        train_loader, 
        val_loader, 
        nn.CrossEntropyLoss(), 
        optim, 
        summary_writer,
        args.full_train,
        device,
        root_dir_val,
    )

    trainer.train(
        epochs=args.epochs, 
        val_frequency=args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency
    )

    trainer.print_per_class_accuracy()
    trainer.save_model_params(Path(log_dir).name)

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
    if args.full_train:
        tb_log_dir_prefix = f'full_CNN_bs={args.batch_size}_lr={args.learning_rate}_run_' + ("freqmask_" if args.freq_mask else "") + ("time_" if args.time_mask else "") + ("double_" if args.double else "") + ("normalised_" if args.normalise else "")
    else:
        tb_log_dir_prefix = f'non_full_CNN_bs={args.batch_size}_lr={args.learning_rate}_run_' + ("freqmask_" if args.freq_mask else "") + ("time_" if args.time_mask else "") + ("double_" if args.double else "") + ("normalised_" if args.normalise else "")

    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main(parser.parse_args())
