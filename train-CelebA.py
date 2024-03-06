import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from dataset import CelebA
from model_celeba import VAE_CelebA
from loss import Loss_VAE
from tqdm import tqdm


DEVICE='cpu'
BATCH_SIZE=64
LEARNING_RATE = 1e-4
EPOCHS = 10

prev_iter = 3
path_checkpoint = f'models/checkpoint-{prev_iter}.pth'
checkpoints = torch.load(path_checkpoint)

dataset = CelebA('data/CELEBA/images/img_align_celeba')
trainloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = VAE_CelebA().to(DEVICE)
model.load_state_dict(checkpoints['model_state_dict'])
criterion = Loss_VAE()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer.load_state_dict(checkpoints['optimizer_state_dict'])

def training():

    print(f"current checkopoint : {path_checkpoint} Loss is {checkpoints['LOSS']}")

    for epoch in range(prev_iter+1, EPOCHS):
        epoch_loss = 0
        
        for images in tqdm(trainloader):
            images = images.to(DEVICE)
            rec_imgs, mu, sigma = model(images)
            loss = criterion(images, rec_imgs, mu, sigma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()

        print(f'For epoch {epoch}, Loss is {epoch_loss}')

        checkpoint = {'EPOCH':epoch,
                      'model_state_dict':model.state_dict(),
                      'optimizer_state_dict':optimizer.state_dict(),
                      'LOSS':epoch_loss}

        torch.save(checkpoint, f"models/checkpoint-{epoch}.pth")

if __name__ == '__main__':
    training()



