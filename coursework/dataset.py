from torch.utils.data.dataset import Dataset
import torch
from torchvision import transforms

import numpy as np
from pathlib import Path
import pandas as pd

class DCASE(Dataset):
    def __init__(
        self, 
        root_dir: str, 
        clip_duration: int, 
        transform=None
    ):
        self._root_dir = Path(root_dir)
        self._labels = pd.read_csv((self._root_dir / 'labels.csv'), names=['file', 'label'])
        self._labels['label'] = self._labels.label.astype('category').cat.codes.astype('int') #create categorical labels
        self.label_dict()
        self._clip_duration = clip_duration
        self._total_duration = 30 #DCASE audio length is 30s
        self._num_clips = self._total_duration // clip_duration 
        self._data_len = len(self._labels)
        self.transform = transform

    def __getitem__(self, index):
        #reading spectrograms
        filename, label = self._labels[['file','label_cat']].iloc[index]
        filepath = self._root_dir / 'audio'/ filename
        spec = torch.from_numpy(np.load(filepath))

        #splitting spec
        spec = self.__trim__(spec)
        
        means = spec.mean()
        stds = spec.std()
        spec = (spec - means) / stds

        if self.transform:
            spec = self.transform(spec)

        return spec, label

    def __trim__(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Trims spectrogram into multiple clips of length specified in self._num_clips
        :param spec: tensor containing spectrogram of full audio signal of shape [1, 60, 1501]
        :return: tensor containing stacked spectrograms of shape [num_clips, 60, clip_length] ([10, 60, 150] with 3s clips)
        """
        time_steps = spec.size(-1)
        self._num_clips = self._total_duration // self._clip_duration
        time_interval = int(time_steps // self._num_clips)
        all_clips = []
        for clip_idx in range(self._num_clips):
            start = clip_idx * time_interval
            end = start + time_interval
            spec_clip = spec[:, start:end]
            #spec_clip = torch.squeeze(spec_clip)
            # print(spec_clip.shape)
            all_clips.append(spec_clip)
            
        specs = torch.stack(all_clips)
        return specs

    def label_dict(self):
        self._labels['label_cat'] = self._labels.label.astype('category').cat.codes.astype('int')
        self._labels['clip_no'] = self._labels.file.apply(lambda file: (file[0], int(file[1:4])))
        self._labels['start_frame'] = self._labels.file.apply(lambda file: file.split('_')[1])
        self._labels['stop_frame'] = self._labels.file.apply(lambda file: file[:-4].split('_')[2])

    def get_num_clips(self) -> int:
        """
        Gets number of clips the raw audio has been split into
        :return: self._num_clips of type int
        """
        return self._num_clips

    def __len__(self):
        return self._data_len

class V_DCASE(Dataset):
    def __init__(self, root_dir: str, clip_duration: int):
        self._root_dir = Path(root_dir)
        self._labels = pd.read_csv((self._root_dir / 'labels.csv'), names=['file', 'label'])
        self._labels['label'] = self._labels.label.astype('category').cat.codes.astype('int') #create categorical labels
        self._clip_duration = clip_duration
        self._total_duration = 30 #DCASE audio length is 30s
        self._num_clips = self._total_duration // clip_duration 
        self._data_len = len(self._labels)

    def __getitem__(self, index):
        #reading spectrograms
        filename, label = self._labels.iloc[index]
        filepath = self._root_dir / 'audio'/ filename
        spec = torch.from_numpy(np.load(filepath))

        #splitting spec
        spec = self.__trim__(spec)
 
        means = spec.mean()
        stds = spec.std()
        spec = (spec - means) / stds

        return spec, label

    def __trim__(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Trims spectrogram into multiple clips of length specified in self._num_clips
        :param spec: tensor containing spectrogram of full audio signal of shape [1, 60, 1501]
        :return: tensor containing stacked spectrograms of shape [num_clips, 60, clip_length] ([10, 60, 150] with 3s clips)
        """
        time_steps = spec.size(-1)
        self._num_clips = self._total_duration // self._clip_duration
        time_interval = int(time_steps // self._num_clips)
        all_clips = []
        for clip_idx in range(self._num_clips):
            start = clip_idx * time_interval
            end = start + time_interval
            spec_clip = spec[:, start:end]
            #spec_clip = torch.squeeze(spec_clip)
            all_clips.append(spec_clip)

        specs = torch.stack(all_clips)
        return specs

    def get_num_clips(self) -> int:
        """
        Gets number of clips the raw audio has been split into
        :return: self._num_clips of type int
        """
        return self._num_clips

    def __len__(self):
        return self._data_len

class NF_DCASE(Dataset):
    
    def __init__(
        self, 
        root_dir: str, 
        clip_duration: int, 
        clip_filter,
        transform = None
    ):
        self._root_dir = Path(root_dir)
        self._labels = pd.read_csv((self._root_dir / 'labels.csv'), names=['file', 'label'])
        self.label_dict()
        self._labels = self._labels[self._labels['file'].isin(clip_filter)]
        self._clip_duration = clip_duration
        self._total_duration = 30 #DCASE audio length is 30s
        self._num_clips = self._total_duration // clip_duration 
        self._data_len = len(self._labels)
        self.transform = transform
        
    def __getitem__(self, index):
        #reading spectrograms
        filename, label = self._labels[['file','label_cat']].iloc[index]
        filepath = self._root_dir / 'audio'/ filename
        spec = torch.from_numpy(np.load(filepath))

        #splitting spec
        spec = self.__trim__(spec)
         
        means = spec.mean()
        stds = spec.std()
        spec = (spec - means) / stds

        if self.transform:
            spec = self.transform(spec)

        return spec, label

    def __trim__(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Trims spectrogram into multiple clips of length specified in self._num_clips
        :param spec: tensor containing spectrogram of full audio signal of shape [1, 60, 1501]
        :return: tensor containing stacked spectrograms of shape [num_clips, 60, clip_length] ([10, 60, 150] with 3s clips)
        """
        time_steps = spec.size(-1)
        self._num_clips = self._total_duration // self._clip_duration
        time_interval = int(time_steps // self._num_clips)
        all_clips = []
        for clip_idx in range(self._num_clips):
            start = clip_idx * time_interval
            end = start + time_interval
            spec_clip = spec[:, start:end]
            #spec_clip = torch.squeeze(spec_clip)
            all_clips.append(spec_clip)

        specs = torch.stack(all_clips)
        return specs
    
    def label_dict(self):
        self._labels['label_cat'] = self._labels.label.astype('category').cat.codes.astype('int')
        self._labels['clip_no'] = self._labels.file.apply(lambda file: (file[0], int(file[1:4])))
        self._labels['start_frame'] = self._labels.file.apply(lambda file: file.split('_')[1])
        self._labels['stop_frame'] = self._labels.file.apply(lambda file: file[:-4].split('_')[2])

    def get_num_clips(self) -> int:
        """
        Gets number of clips the raw audio has been split into
        :return: self._num_clips of type int
        """
        return self._num_clips
    
    def __len__(self):
        return self._data_len
