import torch
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from model import VAE
from collections import Counter
import os

DEVICE = 'mps'

print(os.listdir())
model = torch.load('models/mnist-2d.pkl').to(DEVICE)
model.eval()
dataset = MNIST(root='data/', train=True, transform=ToTensor(), download=True)
max_num = 100

color = {0:'red',
          1:'blue',
          2:'green',
          3:'yellow',
          4:'purple',
          5:'black',
          6:'grey',
          7:'darkgoldenrod',
          8:'midnightblue',
          9:'deeppink'}

digits = {0:[],
          1:[],
          2:[],
          3:[],
          4:[],
          5:[],
          6:[],
          7:[],
          8:[],
          9:[]}

stop = False

for i in range(len(dataset)):
    img, label = dataset[i]
    if len(digits[label])<=max_num:
        digits[label].append(img)
    if all(len(digits[label_j]) > max_num for label_j in range(10)):
        break

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for i in range(10):
    imgs = digits[i]
    for j, img in enumerate(imgs):
        coord = model(img.to(DEVICE).flatten())[1].detach().cpu()
        if j<max_num-1:
            ax.scatter(coord[0], coord[1], c=color[i], alpha=0.6)
        elif j==max_num-1:
            ax.scatter(coord[0], coord[1], c=color[i], alpha=0.6, label=i)
plt.legend()
plt.show()
plt.close()

