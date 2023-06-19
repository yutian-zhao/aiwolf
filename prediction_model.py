import torch
from logloader import AIWolfDataset
from torch.utils.data import DataLoader
from torch import nn
import random

class CNNLSTM(nn.Module):
    def __init__(self, in_channel=8, hid_dim=800):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, 32, 3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.flat_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5*5*64, hid_dim)
        )
        self.lstm = nn.LSTM(hid_dim, hid_dim, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
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

def train_loop(dataloader, model, loss_fn, optimizer, device):
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
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (torch.round(pred) == y).type(torch.float).sum()/torch.numel(y)

    test_loss /= num_batches
    correct /= num_batches
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == '__main__':
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    dataset_dir = "gat2017log15.pt"
    aiwolf_dataset = AIWolfDataset(dataset_dir)
    # dataset specific
    valid_set_indices = [*range(1000)]
    if 'gat2017log15' in dataset_dir:
        invalid_set_indices = [23, 398]
        valid_set_indices = [i for i in valid_set_indices if i not in invalid_set_indices]
    train_set_indices = random.sample(valid_set_indices, k=800)
    test_set_indices = [i for i in valid_set_indices if i not in train_set_indices]
    train_indices = []
    test_indices = []
    for i in range(1000):
        if i < 23:
            if i in train_set_indices:
                train_indices += [(100*i)+j for j in range(100)]
            elif i in test_set_indices:
                test_indices += [(100*i)+j for j in range(100)]
        elif 23 < i < 398:
            if i in train_set_indices:
                train_indices += [(100*i)+j-1 for j in range(100)]
            elif i in test_set_indices:
                test_indices += [(100*i)+j-1 for j in range(100)]
        else:
            if i in train_set_indices:
                train_indices += [(100*i)+j-2 for j in range(100)]
            elif i in test_set_indices:
                test_indices += [(100*i)+j-2 for j in range(100)]
    assert len(set(test_indices+train_indices)) == 99800
    train_dataset = torch.utils.data.Subset(aiwolf_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(aiwolf_dataset, test_indices)
    print("training dataset len: {}; testing dataset len: {}".format(len(train_dataset), len(test_dataset)))
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    model = CNNLSTM().to(device)
    learning_rate = 1e-4
    batch_size = 512
    epochs = 100
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        test_loop(test_dataloader, model, loss_fn, device)
    print("Done!")

