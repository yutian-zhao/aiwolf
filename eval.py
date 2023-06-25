import logging
import torch
from logloader import AIWolfDataset
from torch.utils.data import DataLoader
from torch import nn
import random
import numpy as np
from datetime import datetime
import pandas as pd
from prediction_model import CNNLSTM, test_loop

if __name__ == '__main__':
    start_time = datetime.now()
    pd.set_option("display.precision", 1)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f'Using device: {device}')

    model = torch.load("models/CNNLSTM_0625170355.pt").to(device)
    dataset_name = 'gat2017log15'
    dataset_dir = f"data/{dataset_name}.pt"
    aiwolf_dataset = AIWolfDataset(dataset_dir)
    test_dataloader = DataLoader(aiwolf_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    learning_rate = 1e-4
    batch_size = 64
    epochs = 100
    weight_decay = 1
    ratio = 0.9
    cross_entropy = False
    if cross_entropy:
        weight = torch.tensor([15/8, 15, 15, 15, 15/3, 15])
        weight = weight/torch.sum(weight)
        weight = weight.to(device)
        loss_fn = nn.CrossEntropyLoss(weight=weight, reduction="none")
    else:
        loss_fn = nn.HuberLoss(reduction="none", delta=1.0) # nn.MSELoss(reduction="none") #

    test_loss, test_acc, test_table = test_loop(test_dataloader, model, loss_fn, device, mode='test', cross_entropy=cross_entropy, ratio=ratio)
    print(test_table)
    test_table.to_csv('evals/CNNLSTM_{}_{}.csv'.format(dataset_name, start_time.strftime('%m%d%H%M%S')))
    # start_time = datetime.now()
    # model.eval()
    # test_data = torch.unsqueeze(aiwolf_dataset[0][0], dim=0).to(device)
    # pred = model(test_data)
    # print(pred.shape)
    
    end_time = datetime.now()
    duration = end_time - start_time
    print("Duration: {}".format(str(duration)))



