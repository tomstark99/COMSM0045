# Files

## download_dataset.sh

This is a shell script file that will download and extract the full dataset required for the coursework. It will extract it in such a way so it will work with the `train_cnn.py` file. Into a folder called `ADL_DCASE_DATA/` in the `root` directory (where it is ran). This can be useful for testing the coursework on BC4 or any machine where a shell terminal is the only access and things like `scp` are not possible or slow

# Running the coursework

Running the coursework can be done either directly through an interactive node session on a GPU or through a batch job for which we have provided an sbatch script `train_cnn.sh`

## Interactive Session

Run with any [args](#args) needed for the desired run

```bash
$ python train_cnn.py --full-train --use-cuda
```

## Sbatch script

Edit the `train_cnn.sh` file to add your desired args after which run

```bash
$ sbatch train_cnn.sh
```

you can check the process of your job by checking the queue with your username

```bash
$ watch -n 1 squeue --user <YOUR_USERNAME>
```

## args

| args | Description | Default Value |
| - | - | - |
| `--full-train` | trains the model using full train, if omitted the model is trained using non-full training | False |
| `--learning-rate` | learning rate of the model | 1e-3 |
| `--batch-size` | batch size of the model | 64 |
| `--epochs` | number of epochs the training will be run on, only use this for full training | 50 |
| `--val-frequency` | how frequently to test the model on the validation set in number of epochs | 1 |
| `--use-cuda` | uses the GPU for training | False | 
| `--train-split` | percentage of the training data to split as a validation set | 0.25 |
| `--freq-mask` | probability of frequency mask being applied, use values between 0 and 1 | 0 |
| `--time-mask` | probability of time mask being applied, use values between 0 and 1 | 0 |
| `--double` | applies any active mask augmentation twice | False |
| `--normalise` | normalises training and validaton data to have zero mean | False |

