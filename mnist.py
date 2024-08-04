import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt

x, y = torch.load('/Users/elysia/Coding Projects Repos/MNIST-project-exercise/MNIST/processed/training.pt')

y.shape

plt.imshow(x[2].numpy())
plt.title(f'Number is {y[2].numpy()}')
plt.colorbar()
plt.show()

y_original = torch.tensor([2, 4, 3, 0, 1])
y_new = F.one_hot(y_original)

y_original
y_new
y

y_new = F.one_hot(y, num_classes=10)
y_new.shape

x.shape

x.view(-1,28**2).shape

class CTDataset(Dataset):
    def __init__(self, filepath):
        self.x, self.y = torch.load(filepath)
        self.x = self.x / 255.
        self.y = F.one_hot(self.y, num_classes=10).to(float)
    def __len__(self): 
        return self.x.shape[0]
    def __getitem__(self, ix): 
        return self.x[ix], self.y[ix]
    
train_ds = CTDataset('/Users/elysia/Coding Projects Repos/MNIST-project-exercise/MNIST/processed/training.pt')
test_ds = CTDataset('/Users/elysia/Coding Projects Repos/MNIST-project-exercise/MNIST/processed/test.pt')

len(train_ds)

xs, ys = train_ds[0:4]

ys.shape

train_dl = DataLoader(train_ds, batch_size=5)

for x, y in train_dl:
    print(x.shape)
    print(y.shape)
    break

len(train_dl)

L = nn.CrossEntropyLoss()

class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(28**2,100)
        self.Matrix2 = nn.Linear(100,50)
        self.Matrix3 = nn.Linear(50,10)
        self.R = nn.ReLU()
    def forward(self,x):
        x = x.view(-1,28**2)
        x = self.R(self.Matrix1(x))
        x = self.R(self.Matrix2(x))
        x = self.Matrix3(x)
        return x.squeeze()
    
    
f = MyNeuralNet()

xs.shape
f(xs)
ys
L(f(xs), ys)

def train_model(dl, f, n_epochs=20):
    opt = SGD(f.parameters(), lr=0.01)
    L = nn.CrossEntropyLoss()

    losses = []
    epochs = []
    for epoch in range(n_epochs):
        print(f'Epoch {epoch}')
        N = len(dl)
        for i, (x, y) in enumerate(dl):
            opt.zero_grad() 
            loss_value = L(f(x), y) 
            loss_value.backward() 
            opt.step() 
            epochs.append(epoch+i/N)
            losses.append(loss_value.item())
    return np.array(epochs), np.array(losses)

epoch_data, loss_data = train_model(train_dl, f)

plt.plot(epoch_data, loss_data)
plt.xlabel('Epoch Number')
plt.ylabel('Cross Entropy')
plt.title('Cross Entropy (per batch)')

epoch_data_avgd = epoch_data.reshape(20,-1).mean(axis=1)
loss_data_avgd = loss_data.reshape(20,-1).mean(axis=1)

plt.plot(epoch_data_avgd, loss_data_avgd, 'o--')
plt.xlabel('Epoch Number')
plt.ylabel('Cross Entropy')
plt.title('Cross Entropy (avgd per epoch)')

y_sample = train_ds[0][1]
y_sample

x_sample = train_ds[0][0]
yhat_sample = f(x_sample)
yhat_sample

torch.argmax(yhat_sample)

plt.imshow(x_sample)

xs, ys = train_ds[0:2000]
yhats = f(xs).argmax(axis=1)

fig, ax = plt.subplots(10,4,figsize=(10,15))
for i in range(40):
    plt.subplot(10,4,i+1)
    plt.imshow(xs[i])
    plt.title(f'Predicted Digit: {yhats[i]}')
fig.tight_layout()
plt.show()

xs, ys = test_ds[:2000]
yhats = f(xs).argmax(axis=1)

fig, ax = plt.subplots(10,4,figsize=(10,15))
for i in range(40):
    plt.subplot(10,4,i+1)
    plt.imshow(xs[i])
    plt.title(f'Predicted Digit: {yhats[i]}')
fig.tight_layout()
plt.show()