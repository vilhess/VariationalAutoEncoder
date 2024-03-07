import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from model_celeba import VAE_CelebA
from dataset import CelebA

dataset = CelebA('data/CELEBA/images/img_align_celeba')
DEVICE='cpu'

checkpoints = torch.load('models/checkpoint-8.pth')
parameters = checkpoints['model_state_dict']

model = VAE_CelebA()
model.load_state_dict(parameters)

idxs = torch.randint(0, len(dataset), (5,))
imgs = []
for idx in idxs:
    imgs.append(dataset[idx])
imgs = torch.stack(imgs)
print(imgs.shape)

with torch.no_grad():
    reconstructed = model(imgs.to(DEVICE))[0]

fig = plt.figure(figsize=(9, 4))
for idx, (img, img_rec) in enumerate(zip(imgs, reconstructed)):
    ax = fig.add_subplot(2, 5, idx+1)
    bx = fig.add_subplot(2, 5, idx+6)
    ax.imshow(img.cpu().permute(1, 2, 0), cmap="gray")
    bx.imshow(img_rec.cpu().permute(1, 2, 0), cmap='gray')
    ax.axis('off')
    bx.axis('off')            
plt.show()
plt.close()




