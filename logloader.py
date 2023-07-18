# import numpy
import os
import torch
from torch.utils.data import Dataset

class AIWolfDataset(Dataset):
    def __init__(self, datasets=[]):
        super().__init__()
        data_list = []
        label_list = []
        for dataset in datasets:
            data, labels = torch.load(dataset)
            data_list.append(data)
            label_list.append(labels)

        self.data = torch.cat(data_list, dim=0)
        self.labels = torch.cat(label_list, dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

if __name__ == '__main__':
    aiwolf_dataset = AIWolfDataset(["data/gat2017log15.pt"])
    print(len(aiwolf_dataset))
    # data, label = aiwolf_dataset[0]
    # torch.set_printoptions(profile="full")
    # print(data[1])
    # print(data.shape)
    # print(label)
    # torch.set_printoptions(profile="default")
    # assert torch.sum(data[13]) == 0