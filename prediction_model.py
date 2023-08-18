import logging
import torch
from logloader import AIWolfDataset
from torch.utils.data import DataLoader
from torch import nn
import random
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

class CNNLSTM(nn.Module):
    def __init__(self, in_channel=8, hid_dim=800, dropout=0, cross_entropy=False, bce_loss=False, auxiliary=False, aux_dim=15):
        # cross_entropy: use cross entropy loss
        # bec_loss: use BCELoss
        # auxiliary: whether include loss of auxiliary task (vote prediction)
        super().__init__()
        self.cross_entropy = cross_entropy
        self.bce_loss = bce_loss
        self.auxiliary = auxiliary
        self.aux_dim = aux_dim
        if self.cross_entropy:
            self.out_dim = 6*15
        else:
            self.out_dim = 15

        self.cnn = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, 32, 3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(32, 64, 3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.flat_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5*5*64, hid_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hid_dim, hid_dim, batch_first=True)
        if self.bce_loss:
            self.linear = nn.Sequential(
                nn.ReLU(),
                nn.Linear(hid_dim, hid_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hid_dim, self.out_dim),
                nn.Sigmoid(),
            )
        else:
            self.linear = nn.Sequential(
                nn.ReLU(),
                nn.Linear(hid_dim, hid_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hid_dim, self.out_dim),
            )
        if auxiliary:
            self.aux_linear = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(hid_dim, 100),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(100, self.aux_dim),
                    nn.Sigmoid(),
                )

    def forward(self, x):
        """ Computes the forward pass """
        seq_len = x.shape[1]
        num_channel = x.shape[2]
        x = x.view(-1, num_channel, 15, 15) # (B*L, C, D, D)
        x = self.cnn(x) # (B*L, C, D, D)
        x = self.flat_linear(x) # (B*L, D)
        x = x.view(-1, seq_len, x.shape[-1]) # (B, L, D)
        x, (hn, cn) = self.lstm(x) # (B, L, D)
        if self.auxiliary:
            aux_x = self.aux_linear(x) # (B, L, D)
        x = self.linear(x) # (B, L, D)
        if self.cross_entropy:
            x = x.view(-1, seq_len, 6, 15) # (B, L, C, D)
            # x = x.permute(0, 2, 1, 3).contiguous() # (B, C, L, D)
            softmax_x = torch.nn.functional.softmax(x, dim=2)
            if self.auxiliary:
                return x, aux_x, softmax_x
            else:
                return x, softmax_x
        else: 
            if self.auxiliary:
                return x, aux_x
            else:
                return x

class EarlyStopper:
    def __init__(self, if_save=False, patience=1, min_delta=0):
        # if_save: whether save the best model so far
        # patience: stop training after ```patience``` times iterations
        # min_delta: tolerance when comparing losses
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.if_save = if_save

    def early_stop(self, validation_loss, state_dict=None, path=None):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            if state_dict and self.if_save and path:
                torch.save(state_dict, path)
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def compute_loss_weight(ratio, len):
    # output: [1, ratio, ratio**2, ..., ratio**(len-1)]
    # for cross entropy loss
    output = []
    item = 1
    for i in range(len):
        output.append(item)
        item *= ratio
    output = torch.tensor(output).float()
    output /= torch.sum(output)
    return output

def train_loop(dataloader, model, loss_fn, optimizer, device, writer, epoch, cross_entropy=False, bce_loss=False, ratio=1, pred_role='others', auxiliary=False):
    # writer: tensorboard writer
    # ratio: cross entropy wight ratio
    # pred_role: for BCELoss. "werewolf", "villager"
    assert not (cross_entropy and bce_loss)
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, (y, aux_y)) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        # (B, L, D) / (B, C, L, D) # (B, L, D)
        if cross_entropy:
            if auxiliary:
                pred, aux_pred, _ = model(X)
            else:
                pred, _ = model(X)
        else: 
            if auxiliary:
                pred, aux_pred = model(X)
            else:
                pred = model(X)
        
        if cross_entropy:
            pred = pred.permute(0, 2, 1, 3).contiguous() # (B, C, L, D)
            y = (y - 1).long() # Class indices in the range [0,C)
            L = pred.shape[2] 
            pred_t = pred[:, :, -1, :]
        elif bce_loss:
            y = (y==5).float()
            L = pred.shape[1] 
            pred_t = pred[:,-1, :]
        else:
            L = pred.shape[1] 
            pred_t = pred[:,-1, :]
        y = y.to(device) # (B, D)
        aux_y = aux_y.to(device)
        
        
        loss = loss_fn(pred, torch.unsqueeze(y, 1).expand(-1, L, -1)) # (B, L, D) # pred_t
        loss = torch.sum(loss, dim=-1) # (B, L)
        loss_weight = compute_loss_weight(ratio, L).to(device)
        loss = torch.mean(loss@loss_weight) # (B)

        if auxiliary:
            loss += nn.BCELoss()(aux_pred, aux_y) # TODO: weight

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # logging
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            if cross_entropy:
                pred_arg = torch.argmax(pred_t, dim=1) # y-1
                accuracy = (pred_arg[y!=7] == y[y!=7]).type(torch.float).sum()/torch.numel(y[y!=7]) # ignore index
            elif bce_loss:
                if pred_role == 'werewolf':
                    k = 3
                elif pred_role == 'villager':
                    k = 8
                else:
                    k = 1
                idx_n = torch.arange(pred_t.shape[0]).reshape((-1, 1)).expand(-1, k)
                _, idx = torch.topk(pred_t, 3, dim=-1)
                accuracy = (y[idx_n, idx]==1).type(torch.float).sum()/(pred_t.shape[0]*k)
            else:
                pred_arg = torch.round(pred_t)
                accuracy = (pred_arg == y).type(torch.float).sum()/torch.numel(y)
            logging.info(f"loss: {loss:>7f}  accuracy: {accuracy:>7f}  [{current:>5d}/{size:>5d}]")
            
            writer.add_scalar('Loss/Train', loss,  epoch * len(dataloader) + 
                                batch)
            writer.add_scalar('Accuracy/Train', accuracy, 
                              epoch * len(dataloader) + batch)

    return loss, accuracy

def test_loop(dataloader, model, loss_fn, device, writer=None, epoch=None, mode='valid', cross_entropy=False, bce_loss=False, ratio=1, pred_role="others", auxiliary=False):
    assert not (cross_entropy and bce_loss)
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, accuracy = 0, 0
    aux_accuracy = 0
    table = None
    table2 = None
    result = {}

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, (y, aux_y) in dataloader:
            X = X.to(device)
            # (B, L, D) / (B, C, L, D)
            if cross_entropy:
                if auxiliary:
                    pred, aux_pred, _ = model(X)
                else:
                    pred, _ = model(X)
            else: 
                if auxiliary:
                    pred, aux_pred = model(X)
                else:
                    pred = model(X)
            
            if cross_entropy:
                pred = pred.permute(0, 2, 1, 3).contiguous() # (B, C, L, D)
                y = (y - 1).long() # Class indices in the range [0,C)
                L = pred.shape[2] 
                pred_t = pred[:, :, -1, :]
            elif bce_loss:
                y = (y==5).float()
                L = pred.shape[1] 
                pred_t = pred[:,-1, :]
            else:
                L = pred.shape[1] 
                pred_t = pred[:,-1, :]
            y = y.to(device) # (B, D)
            aux_y = aux_y.to(device)
            
            loss = loss_fn(pred, torch.unsqueeze(y, 1).expand(-1, L, -1))
            loss = torch.sum(loss, dim=-1)
            loss_weight = compute_loss_weight(ratio, L).to(device)
            test_loss += torch.mean(loss@loss_weight)
            # test_loss += loss_fn(pred_t, y).item()
            # compute evaluation stats
            if auxiliary:
                test_loss += nn.BCELoss()(aux_pred, aux_y)
                aux_pred_arg = torch.argmax(aux_pred, dim=-1)
                aux_y_arg = torch.argmax(aux_y, dim=-1)
                aux_accuracy += torch.mean((aux_pred_arg == aux_y_arg).float(), dim=0) # (L)
            if cross_entropy:
                pred_arg = torch.argmax(pred_t, dim=1) # y-1
                accuracy += (pred_arg[y!=7] == y[y!=7]).type(torch.float).sum()/torch.numel(y[y!=7]) # ignore index
                if mode=='test':
                    if not table2 is None:
                        table2 += accuracy_table_2(pred, y, device)
                    else:
                        table2 = accuracy_table_2(pred, y, device)
            elif bce_loss:
                if pred_role == 'werewolf':
                    k = 3
                elif pred_role == 'villager':
                    k = 8
                else:
                    k = 1
                pred_arg = pred_t
                idx_n = torch.arange(pred_t.shape[0]).reshape((-1, 1)).expand(-1, k)
                _, idx = torch.topk(pred_t, 3, dim=-1)
                accuracy += (y[idx_n, idx]==1).type(torch.float).sum()/(pred_t.shape[0]*k)
            else:
                pred_arg = torch.round(pred_t)
                accuracy += (pred_arg == y).type(torch.float).sum()/torch.numel(y)
            if not table is None:
                table += accuracy_table(pred, y, cross_entropy, bce_loss)
            else:
                table = accuracy_table(pred, y, cross_entropy, bce_loss)

    test_loss /= num_batches
    accuracy /= num_batches
    table /= num_batches
    result["test_loss"] = test_loss
    result["accuracy"] = accuracy
    if bce_loss:
        df = pd.DataFrame(data=np.round(100*(table).cpu().detach().numpy(), decimals=1))
    else:
        df = pd.DataFrame(data=np.round(100*(table).cpu().detach().numpy(), decimals=1), index=["VILLAGER", "SEER", "MEDIUM", "BODYGUARD", "WEREWOLF", "POSSESSED", 'mean'], columns=[*range(1, L+1)])
    result['df'] = df
    logging.info(f"{mode} error: accuracy: {(100*accuracy):>0.1f}%, avg loss: {test_loss:>8f} \n")
    if auxiliary:
        aux_accuracy /= num_batches
        logging.info(f"auxiliary accuracy: {aux_accuracy}")
        result['aux_accuracy'] = aux_accuracy
    logging.info(f"sample prediction: {pred_arg[0]}, sample y: {y[0]}")
    logging.info(df)
    logging.info('\n')
    if cross_entropy and mode=='test':
        table2 /= num_batches
        df2 = pd.DataFrame(data=np.round(100*(table2).cpu().detach().numpy(), decimals=1), index=["VILLAGER", "SEER", "MEDIUM", "BODYGUARD", "WEREWOLF", "POSSESSED",], columns=[*range(1, L+1)])
        logging.info(df2)
        logging.info('\n')
        result["df2"] = df2
    # logging.info("{} error: accuracy: {:.1f}%, avg loss: {:8f} \n {:.1f}".format(mode, 100*accuracy, test_loss, 100*table))
    if mode == 'valid':
        writer.add_scalar('Loss/Valid', test_loss, epoch * len(dataloader))
        writer.add_scalar('Accuracy/Valid', accuracy, epoch * len(dataloader))
    return result

def accuracy_table(pred, target, cross_entropy=False, bce_loss=False):
    # computes element-wise accuracy
    # pred: N, L, D; N, C, L, D
    # target: N, 15
    # output: 6, L
    if cross_entropy:
        pred = torch.argmax(pred, dim=1)
    else:
        pred = torch.round(pred)
    L = pred.shape[1]
    if bce_loss:
        output = torch.empty(2, L)
    else:
        output = torch.empty(7, L)
    
    target_N = torch.unsqueeze(target, 1).expand(-1, L, -1)
    ids = torch.unique(target, sorted=True)
    # assert len(ids) == 6
    # assert torch.max(ids) == 5 or torch.max(ids) == 6
    for idx, id in enumerate(ids):
        pred_id = torch.where(target_N==id, pred==id, 0)
        output[idx] = torch.sum(pred_id, dim=[0, -1])/torch.sum(target==id)
    output[-1] = torch.mean(output[:-1], dim=0)
    return output

def accuracy_table_2(input, target, device):
    # compute top 3 recall rate
    # input B C L D; target B D
    # output C L
    result = torch.zeros(input.shape[0], input.shape[1], input.shape[2]).to(device)
    _, idx = torch.topk(input, 1, dim=-1)
    for b in range(input.shape[0]):
        for c in range(input.shape[1]):
            for l in range(input.shape[2]):
                for i in idx[b][c][l]:
                    if c in [0, 4]:
                        result[b][c][l] += (target[b][i]==c).float()/3
                    else:
                        result[b][c][l] = (target[b][i]==c).float()
                        break
    return torch.mean(result, dim=0)



if __name__ == '__main__':
    start_time = datetime.now()
    writer = SummaryWriter(filename_suffix='CNNLSTM')
    pd.set_option("display.precision", 1)
    
    logging.basicConfig(filename='logs/{}.log'.format(start_time.strftime('%m%d%H%M%S')), format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO) # encoding='utf-8', 

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logging.info(f"Using {device} device")

    # load dataset
    dataset_names = {'data/final_game_log.pt': 16013, 'data/gamelog2022-686700.pt':100000, 'data/GAT2018.pt':100000, 'data/cedec2017.pt':10000, 'data/gat2017log15.pt':99998, 'data/gat2017log05.pt':100000,  'data/log_cedec2018.pt':10000, 'data/2019final-log15.pt':10000, 'data/2019final-log05.pt':10000, 'data/ANAC2020Log15.pt':10000, 'data/ANAC2020Log05.pt':10000, } # {'data/gamelog2022-686700.pt':100000, 'data/GAT2018.pt':100000, 'data/cedec2017.pt':10000, 'data/gat2017log15.pt':99998, 'data/gat2017log05.pt':100000,  'data/log_cedec2018.pt':20000, 'data/2019final-log15.pt':10000, 'data/2019final-log05.pt':10000, 'data/ANAC2020Log15.pt':10000, 'data/ANAC2020Log05.pt':10000, } #  
    # dataset_dir = [f"data/{dataset_name}.pt" for dataset_name in dataset_names]
    aiwolf_dataset = AIWolfDataset(dataset_names)
    logging.info("data loaded")
    random.seed(10)
    generator = torch.Generator().manual_seed(42)
    
    train_dataset, test_dataset, valid_dataset = torch.utils.data.random_split(aiwolf_dataset, [0.8, 0.1, 0.1], generator=generator)
    
    logging.info("training dataset len: {}; valid dataset: {}, testing dataset len: {}".format(len(train_dataset), len(valid_dataset), len(test_dataset)))

    learning_rate = 1e-4
    batch_size = 64 # 256
    epochs = 100
    weight_decay = 1
    ratio = 0.9
    cross_entropy = True
    bce_loss = False
    pred_role = "werewolf"
    auxiliary = True
    if cross_entropy:
        weight = torch.tensor([15/3, 15, 15, 0, 30, 15]) # [0, 0, 0, 0, 1, 0] [15/8, 15, 15, 15, 15/3, 15] 
        weight = weight/torch.sum(weight)
        weight = weight.to(device)
        # TODO: None reduction cross entropy loss need check!
        loss_fn = nn.CrossEntropyLoss(weight=weight, ignore_index =7, reduction="none")
    elif bce_loss:
        loss_fn = nn.BCELoss(reduction="none")
    else:
        loss_fn = nn.HuberLoss(reduction="none", delta=1.0) # nn.MSELoss(reduction="none") # 
        

    model = CNNLSTM(cross_entropy=cross_entropy, bce_loss=bce_loss, auxiliary=auxiliary).to(device)
    # writer.add_graph(model, train_dataset[0][0].unsqueeze(0).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, 
                                  weight_decay=weight_decay)
    stopper = EarlyStopper(patience=10, min_delta=-0.001, if_save=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # train and eval
    for t in range(epochs):
        logging.info(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device, writer,
                    epoch=t, cross_entropy=cross_entropy, bce_loss=bce_loss, ratio=ratio, pred_role=pred_role, auxiliary=auxiliary)
        result = test_loop(valid_dataloader, model, loss_fn, device, writer,
                           epoch=t, mode='valid', cross_entropy=cross_entropy, bce_loss=bce_loss, ratio=ratio, pred_role=pred_role, auxiliary=auxiliary)
        if stopper.early_stop(result["test_loss"], model.state_dict(), 'models/CNNLSTM_{}.pt'.format(start_time.strftime('%m%d%H%M%S'))):
            break
    result = test_loop(test_dataloader, model, loss_fn, device, mode='test', cross_entropy=cross_entropy, bce_loss=bce_loss, ratio=ratio, pred_role=pred_role, auxiliary=auxiliary)
    writer.add_hparams({'lr': learning_rate, 'bsize': batch_size, "wdecay":weight_decay}, {'Loss/Test': result["test_loss"], 'Accuracy/Test': result["accuracy"]})
    # test_table.to_csv('evals/CNNLSTM_{}_{}.csv'.format(dataset_name, start_time.strftime('%m%d%H%M%S')))

    end_time = datetime.now()
    duration = end_time - start_time
    logging.info("Duration: {}".format(str(duration)))
    # torch.save(model, 'CNNLSTM_{}.pt'.format(start_time.strftime('%m%d%H%M%S')))
    writer.flush()
    writer.close()
