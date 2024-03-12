import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import scipy.io
import numpy as np
from tqdm import tqdm
from model import VAE
from loss import Loss_VAE


EPOCHS = 200
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
DEVICE = 'mps'

dataset = scipy.io.loadmat('data/FREYFACE/frey_rawface.mat')
dataset = dataset['ff'].T.reshape((-1, 1, 28, 20))
dataset = dataset.astype('float32')/255
dataset = torch.from_numpy(dataset)

trainloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = VAE(input_dim=28*20, hidden_dim=200, z_dim=8).to(DEVICE)
criterion = Loss_VAE()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def training(path):

    for epoch in range(EPOCHS):
        epoch_loss = 0

        for data in tqdm(trainloader):
            x = data
            x = x.to(DEVICE).flatten(start_dim=1)
            x_new, mu, sigma = model(x)
            loss = criterion(x, x_new, mu, sigma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(trainloader)
        print(f'For epoch {epoch}, Loss is {epoch_loss}')
    torch.save(model, path)

if __name__ == '__main__':
    training('models/ff-8d.pkl')



            