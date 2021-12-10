# Files

## download_dataset.sh

This is a shell script file that will download and extract the full dataset required for the coursework. It will extract it in such a way so it will work with the notebooks. Into a folder called "ADL_DCASE_DATA/" in the "coursework/" directory. Can be useful for testing the coursework on BC4 or any machine where a shell terminal is the only access 

## args

| args | Description | Default Value |
| - | - | - |
| `--full-train` | trains the model using full train, if omitted the model is trained using non-full training | None |
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

