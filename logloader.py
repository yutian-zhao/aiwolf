# import numpy
import os
import torch
from torch.utils.data import Dataset
import gc
import copy

num_player = 15
num_channel = 8
MAX_DAY_LENGTH = 14

class AIWolfDataset(Dataset):
    def __init__(self, dataset_dict):
        super().__init__()
        self.dataset_len = 0
        for dataset, limit in dataset_dict.items():
            self.dataset_len += limit

        self.data = torch.empty(self.dataset_len, MAX_DAY_LENGTH, num_channel, num_player, num_player)
        self.labels = torch.empty(self.dataset_len, num_player)
        self.vote_labels = torch.empty(self.dataset_len, MAX_DAY_LENGTH, num_player)
        self.head = 0

        for dataset, limit in dataset_dict.items():
            data, labels, vote_labels = torch.load(dataset)
            self.data[self.head:self.head+limit] = data[:limit]
            self.labels[self.head:self.head+limit] = labels[:limit]
            self.vote_labels[self.head:self.head+limit] = vote_labels[:limit]
            self.head += limit
            del data
            del labels
            del vote_labels
            gc.collect()
            print(f"{dataset} loaded")

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        return self.data[idx], (self.labels[idx], self.vote_labels[idx])

if __name__ == '__main__':
    aiwolf_dataset = AIWolfDataset(["data/gat2017log15.pt"])
    print(len(aiwolf_dataset))
    data, (label, vote_label) = aiwolf_dataset[0]
    # torch.set_printoptions(profile="full")
    # print(data[1])
    print(data.shape)
    print(label)
    print(vote_label)
    # torch.set_printoptions(profile="default")
    # assert torch.sum(data[13]) == 0