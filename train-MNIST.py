import torch
import torch.optim as optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import numpy as np
from tqdm import tqdm
from model import VAE
from loss import Loss_VAE


EPOCHS = 200
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
DEVICE = 'mps'

dataset = MNIST(root='data/', train=True, transform=ToTensor(), download=True)
trainloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = VAE(input_dim=28*28, hidden_dim=100, z_dim=2).to(DEVICE)
criterion = Loss_VAE()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def training(path):

    iter_loss = 0
    best_loss = np.inf
    best_model = model
    best_epoch = 0

    for epoch in range(EPOCHS):
        epoch_loss = 0

        for data in tqdm(trainloader):
            x, labels = data
            x = x.to(DEVICE).flatten(start_dim=1)
            x_new, mu, sigma = model(x)
            loss = criterion(x, x_new, mu, sigma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(trainloader)

        if best_loss > epoch_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            best_model = model
            iter_loss=0
            print('current best model')
        else:
            iter_loss+=1
            print(f'Iteration {iter_loss} with no loss improvement')

        print(f'For epoch {epoch}, Loss is {epoch_loss}')

        if iter_loss>=10:
            print(f'Early Stopping reached : saving model from epoch {best_epoch}')
            break
    print(f'best epoch is {best_epoch} with a loss of {epoch_loss}')
    torch.save(best_model, path)

if __name__ == '__main__':
    training(f'models/mnist-2d.pkl')



            