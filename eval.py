import logging
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
         output_names = ['modelOutput'], # the model's output names 
         # variable length axes 
         # UserWarning: Exporting a model to ONNX with a batch_size other than 1, with a variable length with LSTM can cause an error when running the ONNX model with a different batch size. Make sure to save the model with a batch size of 1, or define the initial states (h0/c0) as inputs of the model
         dynamic_axes={'modelInput' : {1: 'length'}, 'modelOutput' : {1: 'length'}})
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

    name = "CNNLSTM_0625170355"
    model = torch.load(f"models/{name}.pt").to(device)
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

    start_time2 = datetime.now()

    test_data = torch.unsqueeze(aiwolf_dataset[0][0], dim=0).to(device)
    print(test_data.shape)
    pred = model(test_data)
    print(pred.shape)
    print("Single pass time: {}".format(str(datetime.now()-start_time2)))
    
    
    duration = datetime.now() - start_time
    print("Duration: {}".format(str(duration)))

    Convert_ONNX(model, test_data, name)



