import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    Convolutional Neural Network

    Args:
        clip_length:
            length of the clips that the spectogram is split into
        num_clips:
            number of clips the spectogram is split into
    """
    
    def __init__(
        self, 
        clip_length: int, 
        num_clips: int
    ):
        super().__init__()
        self.clip_length = clip_length
        self.num_clips = num_clips

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=128,
            kernel_size=5
        )
        self.batch1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=5
        )
        self.batch2 = nn.BatchNorm2d(256)
        self.pool1 = nn.MaxPool2d(
            kernel_size=5,
            stride=5
        )
        self.pool2 = nn.AdaptiveMaxPool2d((4, None))
        self.fc1 = nn.Linear(256*4*25*num_clips, 15)
        
        self.initialise_layer(self.conv1)
        self.initialise_layer(self.conv2)
        
    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """
        input: [B, num_clips, H, W]

        Pre-processing, change dim from:
            [B, C, H, W] -> [C, B, H, W]
        
        where (C) is the number of segments the clip is split into,
        to simulate batch of number of clip segments and uniary depth
        """
        xs = xs.permute(1,0,2,3)
        xs = self.batch1(self.conv1(xs))
        xs = self.pool1(F.relu(xs))
        xs = self.batch2(self.conv2(xs))
        xs = self.pool2(F.relu(xs))
        """
        Re-shape and flatten back to batch dim of 1:
            [num_clips, X, Y, Z] -> [B, num_clips * X * Y * Z]
            
        where (B) = 1 so you are not left with predicitions for each clip segment,
        just one prediction for the whole clip
        """
        xs = xs.view(1,-1)
        xs = self.fc1(xs)
        return xs
        
    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)
        