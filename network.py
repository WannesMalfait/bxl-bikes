# %% Import what we need
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np

# %% Create a class for loading data


class BikeCountingDataset(Dataset):
    def __init__(self, data_paths, max_rows=None):
        self.data = np.genfromtxt(
            data_paths[0], delimiter=',', skip_header=1, dtype=float, max_rows=max_rows)
        for i in range(1, len(data_paths)):
            self.data += np.genfromtxt(
                data_paths[0], delimiter=',', skip_header=1, dtype=float, max_rows=max_rows)
        # self.data = torch.randn(5000, 2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = torch.as_tensor(self.data[idx][:6], dtype=torch.float32)
        # inputs = torch.tensor(self.data[idx][:2], dtype=torch.float32)
        expected = torch.as_tensor(self.data[idx][6], dtype=torch.float32)
        sample = (inputs, expected)
        return sample


# %% Load the data
print('loading data')
training_data_files = [
    f'./data/training_data/CB02411_{year}.csv' for year in (2019, 2020, 2021)]
training_data = BikeCountingDataset(training_data_files)
test_data = BikeCountingDataset(
    ['./data/training_data/CB02411_2021.csv'])
print('done loading data')


batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(
    training_data, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

# Get cpu or gpu device for training.
device = "cuda"  # if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# %% Define model


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 1),
            nn.Flatten(0, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)
# %% Load previous model
# model.load_state_dict(torch.load("model.pth"))

# %% Define loss, training and test functions
loss_fn = nn.L1Loss()

optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, score = 0, 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred: torch.Tensor = model(X)
            loss = loss_fn(pred, y).item()
            test_loss += loss
            temp = pred.sub(y).div(100.).abs()
            score += torch.sub(1, temp).clamp(0, 1).sum().item()
    test_loss /= size
    score /= size
    print(
        f"Test Error: \n Accuracy: {(100*score):>0.2f}%, Avg loss: {test_loss:>8f} \n")

# %% Train and test the model!


epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)

# %% Save the model
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")


# %% Test with new custom input
model.eval()
days = ["Mon", "Tue", "Wen", "Thu", "Fri", "Sat", "Sun"]
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
x = torch.tensor([13, 19.0, 11.5, 6, 3, 105634], dtype=torch.float32)
X = torch.broadcast_to(x, (batch_size, 6)).to(device)
with torch.no_grad():
    pred = model(X)
    print(f'hour {x[0]}; temperature: {x[1]} °C; windspeed: {x[2]} knots; weekday: {days[int(x[3])]}; month: {months[int(x[4])]}; yearcount: {x[5]}')
    print('prediction:', pred[0].item())

# %% Done!
