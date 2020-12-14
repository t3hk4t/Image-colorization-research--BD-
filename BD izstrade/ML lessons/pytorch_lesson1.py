import torch
import numpy as np
import matplotlib
import torch.utils.data
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
matplotlib.use('TkAgg')


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=4, out_features=4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=4, out_features=4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=4, out_features=3),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.layers.forward(x)
        return out


class LossCrossEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, y_prim):
        return -torch.sum(y * torch.log(y_prim + 1e-20))


class LossBinaryCrossEntropy(torch.nn.Module):
    def __init__(self, weights = [1.0, 1.0]):
        super().__init__()
        self.weights = weights

    def forward(self, y, y_prim):
        return -torch.sum(self.weights[0]*y * torch.log(y_prim + 1e-20) +
                          self.weights[1]*(1.0 - y) * torch.log((1.0-y_prim)+1e-20)
                          )


X, Y = load_iris(
    return_X_y=True
)

idxes_rand = np.random.permutation(len(Y))
X = X[idxes_rand]
Y = Y[idxes_rand]
X_max = np.max(X, axis=0)
X_min = np.min(X, axis=0)
X = (((X - X_min) / (X_max - X_min)) - 0.5) * 2.0


class DatasetIris(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.samples = list(zip(X, Y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        np_x, y_idx = self.samples[idx]
        x = torch.FloatTensor(np_x)

        prob = 0.0
        if y_idx == 0:
            prob = 1.0
        np_y = np.array([prob])
        y = torch.FloatTensor(np_y)
        return x, y


data_loader_train = torch.utils.data.DataLoader(
    dataset=DatasetIris(X, Y),
    batch_size=10,
    shuffle=True
)

model = Model()
loss_func = LossBinaryCrossEntropy(weights=[1.0, 2.0])
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

losses_train = []
matric_acc = []

for epoch in range(1, 100):
    plt.clf()
    losses_epoch = []
    acc_epoch = []

    for x, y in data_loader_train:
        y_prim = model.forward(x)
        loss = loss_func.forward(y, y_prim)
        losses_epoch.append(loss.item())  # Tensor(0.1) => 0.1f

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        np_y_prim = y_prim.data.numpy()
        np_y = y.data.numpy()
        acc = np.average(np.equal((np_y == 1.0), (np_y_prim > 0.5)) * 1.0)
        acc_epoch.append(acc)

    loss = np.mean(losses_epoch)
    acc = np.mean(acc_epoch)
    matric_acc.append(acc)
    losses_train.append(loss)
    print(f'Epoch: {epoch} loss: {loss} acc: {acc}')
    plt1 = plt.plot(losses_train, '-b', label='loss')
    ax = plt.twinx()
    plt2 = plt.plot(matric_acc, '--r', label='acc')
    plts = plt1 + plt2
    plt.legend(plts, [it.get_label() for it in plts])
    plt.draw()
    plt.pause(0.1)
input('Quit?')