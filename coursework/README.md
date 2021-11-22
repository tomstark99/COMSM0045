# Files

## download_dataset.sh

This is a shell script file that will download and extract the full dataset required for the coursework. It will extract it in such a way so it will work with the notebooks. Into a folder called "ADL_DCASE_DATA/" in the "coursework/" directory. Can be useful for testing the coursework on BC4 or any machine where a shell terminal is the only access 

## ADL_DCASE_DATA.zip

This is the zip file of the DCASE 2016 dataset. Unzipped it has the following file format:

```bash
$ tree -L 2 .
.
|-- development
|   |-- audio
|   `-- labels.csv
`-- evaluation
    |-- audio
    `-- labels.csv

4 directories, 2 files
```

Each "audio/" directory contains all of the data for that split. The data is stored as arrays where each array represents a spectrogram. The spectrograms have been created according to the parameters described in the paper. "labels.csv" contains the labels for each audio sample inside "audio/". You should exclusively use the data in "Development/" for training. The data in "Evaluation/" is exclusively for evaluating your model. Do not train your model on the data in "Evaluation/".

## dataset.py

This is a PyTorch Dataset implemention for the DCASE 2016 dataset. This code loads the spectrograms provided and splits them into shorter sequences. The DCASE class requires a path to your dataset and the length of your audio clips in seconds. You should use this class in conjuction with a PyTorch DataLoader. You can see examples of how to use a DataLoader in your lab code.

This dataset class will return tensors of the shape [batches, num_clips, height, width]. Most CNN models will expect data in the form [batches, channels, height, width]. In this case there is an additional dimension (num_clips) as a result of the sequence splitting described in the paper. In order to resolve this, we suggest you combine the number of clips into the batch dimension using torch.view(). You can then retrieve the correct dimensions by reshaping your data after passing it through the model. The DCASE class has a function, get_num_clips, which you can call. This function will return the number of clips each spectrogram is split into (determined by the clip length). You may find this useful when reshaping your tensors.

When getting the data from the data loader you have the data in the form:

[B, X, H, W]

Where X is num_clips. CNNs expect a channel dimension though, so you’ll have to merge the first 2 dimensions while retaining all four in general I.e. your shape then becomes:

[B * X, C, H, W]

Where C=1. That’s for the input of the model and you can use torch.view() to do this. However, you then need to account for reshaping the output.

The output will initially be some shape that I can’t remember off the top of my head, but you’ll want to reshape it to:

[B, X, 15]

Where X is the number of clips and 15 is the number of classes. Then you’ll want to take the mean across the clips I.E. dimension 1 and then you’ll end up with the desired:

[B, 15]

## VisualiseSpec.ipynb

This is a jupyter notebook that allows you to visualise a spectrogram. The spectrogram producing code is the same as in dataset.py, and the current parameters match those in the paper. You can adjust these parameters if you wish. 

