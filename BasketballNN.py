import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F, CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class Stats(Dataset):
    def __init__(self):
        # Dataset
        # https://www.kaggle.com/datasets/drgilermo/nba-players-stats
        df = pd.read_csv("stats_basic.csv")
        #self.valid_positions = ['PG', 'PF', 'SF', 'C', 'SG']
        self.valid_positions = ['PG', 'SF', 'C']
        # Dropping players who are labeled not one of the five core positions
        df = df[df['Pos'].isin(self.valid_positions)]
        #print(df.info())


        self.X = torch.tensor(df.iloc[:, 6:-1].to_numpy(), dtype=torch.float)

        #The labels are categorical so assigning each label a numeric value
        labels, uniques = pd.factorize(df.iloc[:, 3])

        self.y = torch.tensor(labels, dtype=torch.long)
        self.len = len(df)

        #Doing a train test split for a validation set for the neural network
        self.X, self.X_valid, self.y, self.y_valid = train_test_split(self.X, self.y, test_size=0.3)
        self.len = len(self.X)


    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.len

    def to_numpy(self):
        return np.array(self.X), np.array(self.y)


class Position(nn.Module):
    def __init__(self):
        # Call the constructor of the super class
        super(Position, self).__init__()

        #Normalizing the data of the features
        self.norm = nn.BatchNorm1d(21)

        # 5 hidden layer neural network
        self.in_to_h1 = nn.Linear(21, 50)
        self.h1_to_h2 = nn.Linear(50, 25)
        self.h2_to_h3 = nn.Linear(25, 15)
        self.h3_to_h4 = nn.Linear(15, 10)
        self.h4_to_out = nn.Linear(10, 3)

    def forward(self, x):
        x = self.norm(x)
        x = F.relu(self.in_to_h1(x))
        x = F.relu(self.h1_to_h2(x))
        x = F.relu(self.h2_to_h3(x))
        x = F.relu(self.h3_to_h4(x))
        return self.h4_to_out(x)


def trainNN(epochs=10, batch_size=32, lr=0.001):
    # Load the Dataset
    bs = Stats()

    # Create data loader
    data_loader = DataLoader(bs, batch_size=batch_size, drop_last=True, shuffle=True)

    # Create an instance of the NN
    pos = Position()

    # loss function
    loss_fn = CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(pos.parameters(), lr=lr)

    running_loss = 0.0
    for epoch in range(epochs):
        for _, data in enumerate(data_loader, 0):
            pos.train()
            x, y = data

            optimizer.zero_grad()

            output = pos(x)

            loss = loss_fn(output, y)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        with torch.no_grad():
            pos.eval()
            print(f"Running loss for epoch {epoch + 1} of {epochs}: {running_loss:.4f}")
            predictions = torch.argmax(pos(bs.X), dim=1)
            correct = (predictions == bs.y).sum().item()
            print(f"Accuracy on train set: {correct / len(bs.X):.4f}")
            predictions = torch.argmax(pos(bs.X_valid), dim=1)
            correct = (predictions == bs.y_valid).sum().item()
            print(f"Accuracy on validation set: {correct / len(bs.X_valid):.4f}")
            pos.train()
        running_loss = 0.0

    cm = confusion_matrix(bs.y_valid, predictions)
    disp_cm = ConfusionMatrixDisplay(cm, display_labels=bs.valid_positions)
    disp_cm.plot()
    plt.show()
    return pos



trainNN(epochs=100)
