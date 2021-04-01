from torch._C import dtype
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np


class BikeCountingDataset(Dataset):
    def __init__(self, data_path, max_rows=None):
        self.data = np.genfromtxt(
            data_path, delimiter=',', skip_header=1, dtype=float, max_rows=max_rows)
        # self.data = torch.randn(5000, 2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = torch.as_tensor(self.data[idx][:6], dtype=torch.float32)
        # inputs = torch.tensor(self.data[idx][:2], dtype=torch.float32)
        expected = torch.as_tensor(self.data[idx][6], dtype=torch.float32)
        sample = (inputs, expected)
        return sample


print('loading data')
training_data = BikeCountingDataset('./data/training_data/CB02411_2020.csv')
test_data = BikeCountingDataset(
    './data/training_data/CB02411_2020.csv', max_rows=1024)
print('done loading data')


batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(
    training_data, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

# Get cpu or gpu device for training.
device = "cuda"  # if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model


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
# model.load_state_dict(torch.load("model.pth"))
print(model)


# def loss_fn(pred: torch.Tensor, y: torch.Tensor):
#     temp = pred.sub(y).div(200.).square()
#     return torch.sub(1, temp).clamp(0, 1).sum().div(float(batch_size))


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


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

model.eval()
x = torch.tensor([13, 30, 2, 6, 3, 114000], dtype=torch.float32)
x = torch.broadcast_to(x, (batch_size, 6)).to(device)
# print(x)
with torch.no_grad():
    pred = model(x)
    print('prediction:', pred[0].item())
# for X, y in test_dataloader:
#     X, y = X.to(device), y.to(device)

#     print('Input:', X[0])

#     pred: torch.Tensor = model.forward(X[0])
#     print('Prediction:', pred, 'expected', y)
#     break
# model.eval()
# with torch.no_grad():
#     pred = model(x)
#     predicted, actual = classes[pred[0].argmax(0)], classes[y]
#     print(f'Predicted: "{predicted}", Actual: "{actual}"')
