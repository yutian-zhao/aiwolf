import logging
import torch
from logloader import AIWolfDataset
from torch.utils.data import DataLoader
from torch import nn
import random
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class CNNLSTM(nn.Module):
    def __init__(self, in_channel=8, hid_dim=800, dropout=0.2):
        super().__init__()
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
            nn.Linear(hid_dim, 15),
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
        x = self.linear(x[:, -1, :])

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

def train_loop(dataloader, model, loss_fn, optimizer, device, writer, epoch):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            accuracy = (torch.round(pred) == y).type(torch.float).sum()/torch.numel(y)
            logging.info(f"loss: {loss:>7f}  accuracy: {accuracy:>7f}  [{current:>5d}/{size:>5d}]")
            
            writer.add_scalar('Loss/Train', loss,  epoch * len(dataloader) + 
                                batch)
            writer.add_scalar('Accuracy/Train', accuracy, 
                              epoch * len(dataloader) + batch)

    return loss, accuracy


def test_loop(dataloader, model, loss_fn, device, writer=None, epoch=None, mode='valid'):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, accuracy = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            accuracy += (torch.round(pred) == y).type(torch.float).sum()/torch.numel(y)

    test_loss /= num_batches
    accuracy /= num_batches
    logging.info(f"{mode} error: accuracy: {(100*accuracy):>0.1f}%, avg loss: {test_loss:>8f} \n")
    if mode == 'valid':
        writer.add_scalar('Loss/Valid', test_loss, epoch * len(dataloader))
        writer.add_scalar('Accuracy/Valid', accuracy, epoch * len(dataloader))
    return test_loss, accuracy

if __name__ == '__main__':
    start_time = datetime.now()
    writer = SummaryWriter(filename_suffix='CNNLSTM')
    
    logging.basicConfig(filename='logs/{}.log'.format(start_time.strftime('%m%d%H%M%S')), format='%(asctime)s [%(levelname)s] %(message)s', encoding='utf-8', level=logging.INFO)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logging.info(f"Using {device} device")

    dataset_dir = "gat2017log15.pt"
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

    model = CNNLSTM().to(device)
    writer.add_graph(model, train_dataset[0][0].unsqueeze(0).to(device))
    learning_rate = 1e-4
    batch_size = 64
    epochs = 100
    weight_decay = 1
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, 
                                  weight_decay=weight_decay)
    stopper = EarlyStopper(patience=10, min_delta=-0.001)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    for t in range(epochs):
        logging.info(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device, writer,
                    epoch=t, )
        valid_loss, valid_acc = test_loop(valid_dataloader, model, loss_fn, device, writer, epoch=t, mode='valid')
        if stopper.early_stop(valid_loss):
            break
    test_loss, test_acc = test_loop(test_dataloader, model, loss_fn, device, mode='test')
    writer.add_hparams({'lr': learning_rate, 'bsize': batch_size, "wdecay":weight_decay}, {'Loss/Test': test_loss, 'Accuracy/Test': test_acc})

    end_time = datetime.now()
    duration = end_time - start_time
    logging.info("Duration: {}".format(str(duration)))

    torch.save(model, 'CNNLSTM_{}.pt'.format(start_time.strftime('%m%d%H%M%S')))
    writer.flush()
    writer.close()
