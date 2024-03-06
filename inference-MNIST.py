import torch
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

DEVICE = 'mps'


model = torch.load('models/mnist.pkl').to(DEVICE)
dataset = MNIST(root='data/', train=True, transform=ToTensor(), download=True)
idxs = torch.randint(0, len(dataset), (10,))
imgs = []
for idx in idxs:
    imgs.append(dataset[idx][0].flatten())
imgs = torch.stack(imgs)

with torch.no_grad():
    reconstructed = model(imgs.to(DEVICE))[0]

fig = plt.figure(figsize=(9, 4))
for idx, (img, img_rec) in enumerate(zip(imgs, reconstructed)):
    ax = fig.add_subplot(2, 10, idx+1)
    bx = fig.add_subplot(2, 10, idx+11)
    ax.imshow(img.cpu().reshape(28, 28, 1), cmap="gray")
    bx.imshow(img_rec.cpu().reshape(28, 28, 1), cmap='gray')
plt.show()
