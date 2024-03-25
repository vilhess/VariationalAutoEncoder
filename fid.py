import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance

from model_celeba import VAE_CelebA
from dataset import CelebA
from model_celeba import VAE_CelebA


dataset = CelebA('data/CELEBA/images/img_align_celeba')
DEVICE='cpu'

model = VAE_CelebA().to(DEVICE)

imgs = []
print('loading dataset...')
for i, img in enumerate(dataset):
    imgs.append(img)
    if i==1000:
        break
imgs = torch.stack(imgs)
print('dataset loaded')

scores = []

checkpoints = [torch.load(f'models/checkpoint-{epoch}.pth', map_location=torch.device(DEVICE)) for epoch in range(30)]
parameters = [checkpoint['model_state_dict'] for checkpoint in checkpoints]


for epoch in range(30):

    model.load_state_dict(parameters[epoch])

    model.eval()

    fid = FrechetInceptionDistance(feature=64)

    print('model inference...')

    with torch.no_grad():
        reconstructeds = model(imgs.to(DEVICE))[0]

    fid.update((imgs*255).to(torch.uint8), real=True)
    fid.update((reconstructeds*255).to(torch.uint8), real=False)

    score = fid.compute()
    print(f'FID score for epoch {epoch} is {score}')
    scores.append(score)

fig = plt.figure()
plt.plot(scores)
plt.ylabel('FID score')
plt.xlabel('epoch')
plt.show()
plt.close