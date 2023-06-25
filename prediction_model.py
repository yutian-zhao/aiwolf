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
    def __init__(self, in_channel=8, hid_dim=800, dropout=0.2, cross_entropy=False):
        super().__init__()
        self.cross_entropy = cross_entropy
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
            nn.Linear(5*5*64, hid_dim)
        )
        self.lstm = nn.LSTM(hid_dim, hid_dim, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, self.out_dim),
        )

    def forward(self, x):
        """ Computes the forward pass """
        seq_len = x.shape[1]
        num_channel = x.shape[2]
        x = x.view(-1, num_channel, 15, 15)
        x = self.cnn(x)
        x = self.flat_linear(x)
        x = x.view(-1, seq_len, x.shape[-1])
        x, (hn, cn) = self.lstm(x)
        x = self.linear(x) # output on the last day
        if self.cross_entropy:
            x = x.view(-1, -1, 6, 15)

        return x

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def compute_loss_weight(ratio, len):
    output = []
    item = 1
    for i in range(len):
        output.append(item)
        item *= ratio
    output = torch.tensor(output).float()
    output /= torch.sum(output)
    return output

def train_loop(dataloader, model, loss_fn, optimizer, device, writer, epoch, cross_entropy=False, ratio=1):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        if cross_entropy:
            y = (y - 1).long()
        y = y.to(device)
        pred = model(X)
        L = pred.shape[1]
        pred_t = pred[:,-1, :]
        loss = loss_fn(pred, torch.unsqueeze(y, 1).expand(-1, L, -1)) # pred_t
        # logging.info(f"Loss Shape: {loss.shape}")
        loss = torch.sum(loss, dim=-1)
        loss_weight = compute_loss_weight(ratio, L).to(device)
        loss = torch.mean(loss@loss_weight)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            if cross_entropy:
                pred_arg = torch.argmax(pred_t, dim=1)
            else:
                pred_arg = torch.round(pred_t)
            accuracy = (pred_arg == y).type(torch.float).sum()/torch.numel(y)
            logging.info(f"loss: {loss:>7f}  accuracy: {accuracy:>7f}  [{current:>5d}/{size:>5d}]")
            
            writer.add_scalar('Loss/Train', loss,  epoch * len(dataloader) + 
                                batch)
            writer.add_scalar('Accuracy/Train', accuracy, 
                              epoch * len(dataloader) + batch)

    return loss, accuracy

def test_loop(dataloader, model, loss_fn, device, writer=None, epoch=None, mode='valid', cross_entropy=False, ratio=1):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, accuracy = 0, 0
    table = None

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            if cross_entropy:
                y = (y - 1).long()
            y = y.to(device)
            pred = model(X)
            L = pred.shape[1]
            pred_t = pred[:,-1, :]
            loss = loss_fn(pred, torch.unsqueeze(y, 1).expand(-1, L, -1))
            loss = torch.sum(loss, dim=-1)
            loss_weight = compute_loss_weight(ratio, L).to(device)
            test_loss += torch.mean(loss@loss_weight)
            # test_loss += loss_fn(pred_t, y).item()
            if cross_entropy:
                pred_ = torch.argmax(pred_t, dim=1)
            else:
                pred_ = torch.round(pred_t)
            accuracy += (pred_ == y).type(torch.float).sum()/torch.numel(y)
            if not table is None:
                table += accuracy_table(pred, y)
            else:
                table = accuracy_table(pred, y)

    test_loss /= num_batches
    accuracy /= num_batches
    table /= num_batches
    df = pd.DataFrame(data=np.round(100*(table).cpu().detach().numpy(), decimals=1), index=["VILLAGER", "SEER", "MEDIUM", "BODYGUARD", "WEREWOLF", "POSSESSED", 'mean'], columns=[*range(1, L+1)])
    logging.info(f"{mode} error: accuracy: {(100*accuracy):>0.1f}%, avg loss: {test_loss:>8f} \n")
    logging.info(df)
    logging.info('\n')
    # logging.info("{} error: accuracy: {:.1f}%, avg loss: {:8f} \n {:.1f}".format(mode, 100*accuracy, test_loss, 100*table))
    if mode == 'valid':
        writer.add_scalar('Loss/Valid', test_loss, epoch * len(dataloader))
        writer.add_scalar('Accuracy/Valid', accuracy, epoch * len(dataloader))
    return test_loss, accuracy, df


def accuracy_table(pred, target):
    # pred: N, L, D
    # target: N, 15
    # output: 6, L
    L = pred.shape[1]
    output = torch.empty(7, L)
    pred = torch.round(pred)
    target_N = torch.unsqueeze(target, 1).expand(-1, L, -1)
    ids = torch.unique(target, sorted=True)
    assert len(ids) == 6
    assert torch.max(ids) == 5 or torch.max(ids) == 6
    for idx, id in enumerate(ids):
        pred_id = torch.where(target_N==id, pred==id, 0)
        output[idx] = torch.sum(pred_id, dim=[0, -1])/torch.sum(target==id)
    output[-1] = torch.mean(output[:-1], dim=0)
    return output

if __name__ == '__main__':
    start_time = datetime.now()
    writer = SummaryWriter(filename_suffix='CNNLSTM')
    pd.set_option("display.precision", 1)
    
    logging.basicConfig(filename='logs/{}.log'.format(start_time.strftime('%m%d%H%M%S')), format='%(asctime)s [%(levelname)s] %(message)s', encoding='utf-8', level=logging.INFO)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logging.info(f"Using {device} device")

    dataset_name = 'gat2017log15'
    dataset_dir = f"data/{dataset_name}.pt"
    aiwolf_dataset = AIWolfDataset(dataset_dir)
    random.seed(10)
    # dataset specific
    indices = [*range(1000)]
    if 'gat2017log15' in dataset_dir:
        invalid_set_indices = [23, 398]
        indices = [i for i in indices if i not in invalid_set_indices]
    
    train_set_indices = random.sample(indices, k=800)
    valid_set_indices = [i for i in indices if i not in train_set_indices]
    test_set_indices = random.sample(valid_set_indices, k=int(len(valid_set_indices)/2))
    valid_set_indices = [i for i in valid_set_indices if i not in
                        test_set_indices]
    train_indices = []
    valid_indices = []
    test_indices = []
    for i in range(1000):
        if i < 23:
            if i in train_set_indices:
                train_indices += [(100*i)+j for j in range(100)]
            elif i in test_set_indices:
                test_indices += [(100*i)+j for j in range(100)]
            elif i in valid_set_indices:
                valid_indices += [(100*i)+j for j in range(100)]
        elif 23 < i < 398:
            if i in train_set_indices:
                train_indices += [(100*i)+j-1 for j in range(100)]
            elif i in test_set_indices:
                test_indices += [(100*i)+j-1 for j in range(100)]
            elif i in valid_set_indices:
                valid_indices += [(100*i)+j-1 for j in range(100)]
        else:
            if i in train_set_indices:
                train_indices += [(100*i)+j-2 for j in range(100)]
            elif i in test_set_indices:
                test_indices += [(100*i)+j-2 for j in range(100)]
            elif i in valid_set_indices:
                valid_indices += [(100*i)+j-2 for j in range(100)]
    assert len(set(test_indices+train_indices+valid_indices)) == 99800, "got {}".format(len(set(test_indices+train_indices+valid_indices)))
    
    train_dataset = torch.utils.data.Subset(aiwolf_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(aiwolf_dataset, test_indices)
    valid_dataset = torch.utils.data.Subset(aiwolf_dataset, valid_indices)
    logging.info("training dataset len: {}; valid dataset: {}, testing dataset len: {}".format(len(train_dataset), len(valid_dataset), len(test_dataset)))

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
        # TODO: None reduction cross entropy loss need check!
        loss_fn = nn.CrossEntropyLoss(weight=weight, reduction="none")
    else:
        loss_fn = nn.HuberLoss(reduction="none", delta=1.0) # nn.MSELoss(reduction="none") # 

    model = CNNLSTM(cross_entropy=cross_entropy).to(device)
    # writer.add_graph(model, train_dataset[0][0].unsqueeze(0).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, 
                                  weight_decay=weight_decay)
    stopper = EarlyStopper(patience=10, min_delta=-0.001)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    for t in range(epochs):
        logging.info(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device, writer,
                    epoch=t, cross_entropy=cross_entropy, ratio=ratio)
        valid_loss, _ = test_loop(valid_dataloader, model, loss_fn, device, writer, epoch=t, mode='valid', cross_entropy=cross_entropy, ratio=ratio)
        if stopper.early_stop(valid_loss):
            break
    test_loss, test_acc, test_table = test_loop(test_dataloader, model, loss_fn, device, mode='test', cross_entropy=cross_entropy, ratio=ratio)
    writer.add_hparams({'lr': learning_rate, 'bsize': batch_size, "wdecay":weight_decay}, {'Loss/Test': test_loss, 'Accuracy/Test': test_acc})
    test_table.to_csv('evals/CNNLSTM_{}_{}.csv'.format(dataset_name, start_time.strftime('%m%d%H%M%S')))

    end_time = datetime.now()
    duration = end_time - start_time
    logging.info("Duration: {}".format(str(duration)))

    torch.save(model, 'CNNLSTM_{}.pt'.format(start_time.strftime('%m%d%H%M%S')))
    writer.flush()
    writer.close()
