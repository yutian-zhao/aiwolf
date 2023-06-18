# import numpy
import os
import torch
from torchvision import datasets

class AIWolfDataset(Dataset):
    def __init__(self, dir):
        super().__init__()
        self.dir = dir
        self.data = torch.load(dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return data[idx], label
