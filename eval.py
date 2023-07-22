import logging
import os
import torch
import torch.onnx 
from logloader import AIWolfDataset
from torch.utils.data import DataLoader
from torch import nn
import random
import numpy as np
from datetime import datetime
import pandas as pd
from prediction_model import CNNLSTM, test_loop
import json

#Function to Convert to ONNX 
def Convert_ONNX(model, dummy_input, name): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    # dummy_input = torch.randn(1, input_size, requires_grad=True)  

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         name+'.onnx',       # where to save the model  
         export_params=True,  # default, store the trained parameter weights inside the model file 
        #  opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # default, whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['pred', "aux_pred", 'softmax_pred'], # the model's output names 
         # variable length axes 
         # UserWarning: Exporting a model to ONNX with a batch_size other than 1, with a variable length with LSTM can cause an error when running the ONNX model with a different batch size. Make sure to save the model with a batch size of 1, or define the initial states (h0/c0) as inputs of the model
         dynamic_axes={'modelInput' : {1: 'length'}, 'pred' : {1: 'length'}, 'aux_pred': {1: 'length'}, 'softmax_pred': {1: 'length'}})
    print(" ") 
    print('Model has been converted to ONNX') 

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

    learning_rate = 1e-4
    batch_size = 64
    epochs = 100
    weight_decay = 1
    ratio = 0.9
    cross_entropy = True
    bce_loss = False
    pred_role = "werewolf"
    auxiliary = True
    
    if cross_entropy:
        weight = torch.tensor([15/4, 15, 15, 5, 25, 15]) # [15/8, 15, 15, 15, 15/3, 15] [15/4, 15, 15, 0, 30, 15]
        weight = weight/torch.sum(weight)
        weight = weight.to(device)
        # TODO: None reduction cross entropy loss need check!
        loss_fn = nn.CrossEntropyLoss(weight=weight, reduction="none")
    elif bce_loss:
        loss_fn = nn.BCELoss(reduction="none")
    else:
        loss_fn = nn.HuberLoss(reduction="none", delta=1.0) # nn.MSELoss(reduction="none") #

    name = "0722172447"
    model = CNNLSTM(cross_entropy=cross_entropy, bce_loss=bce_loss, auxiliary=auxiliary).to(device)
    model.load_state_dict(torch.load(f"models/CNNLSTM_{name}.pt"))
    model.eval()

    # model = torch.load(f"models/{name}.pt")

    # state_dict = model.state_dict()
    # model = CNNLSTM(cross_entropy=cross_entropy, bce_loss=bce_loss).to(device)
    # model.load_state_dict(state_dict)
    # model.eval()

    dataset_name = 'log' # temp_dataset
    dataset_dir = f"data/{dataset_name}.pt"
    aiwolf_dataset = AIWolfDataset({dataset_dir:100})
    test_dataloader = DataLoader(aiwolf_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    
    result = test_loop(test_dataloader, model, loss_fn, device, mode='others', cross_entropy=cross_entropy, bce_loss=bce_loss, ratio=ratio, pred_role=pred_role, auxiliary=auxiliary) # test
    json_result = {}
    for k, v in result.items():
        if type(v) is torch.Tensor:
            json_result[k] = v.tolist()
        elif type(v) is pd.DataFrame:
            json_result[k] = v.to_string()
        else:
            json_result[k] = v
    # json_result = {k:(v.tolist() if type(v) is torch.Tensor else v) for k, v in result.items()}
    with open('evals/{}_{}.json'.format(name, dataset_name), "w") as f:
        json.dump(json_result, f, indent=4)

    # print("test table: ", test_table)
    # print("test table2: ", test_table2)
    # print("test loss: ", test_loss)
    # print("test accuracy: ", test_acc)
    # test_table.to_csv('evals/{}_{}_{}.csv'.format(name, dataset_name, start_time.strftime('%m%d%H%M%S')))

    start_time2 = datetime.now()

    test_data = torch.unsqueeze(aiwolf_dataset[0][0], dim=0).to(device)
    # print(test_data.shape)
    pred = model(test_data)
    # print(pred.shape)
    print("Single pass time: {}".format(str(datetime.now()-start_time2)))
    
    
    duration = datetime.now() - start_time
    print("Duration: {}".format(str(duration)))

    # Convert_ONNX(model, test_data, f"CNNLSTM_{name}")



