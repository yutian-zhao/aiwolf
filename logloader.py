# import numpy
import os
import torch
from torch.utils.data import Dataset

class AIWolfDataset(Dataset):
    def __init__(self, dir):
        super().__init__()
        self.dir = dir
        self.data, self.labels = torch.load(dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

if __name__ == '__main__':
    aiwolf_dataset = AIWolfDataset("data/gat2017log15.pt")
    print(len(aiwolf_dataset))
    # data, label = aiwolf_dataset[0]
    # torch.set_printoptions(profile="full")
    # print(data[1])
    # print(data.shape)
    # print(label)
    # torch.set_printoptions(profile="default")
    # assert torch.sum(data[13]) == 0