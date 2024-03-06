import torch
import matplotlib.pyplot as plt
import scipy.io

DEVICE = 'mps'

dataset = scipy.io.loadmat('data/FREYFACE/frey_rawface.mat')
dataset = dataset['ff'].T.reshape((-1, 1, 28, 20))
dataset = dataset.astype('float32')/255
dataset = torch.from_numpy(dataset)

model = torch.load('models/ff-8d.pkl').to(DEVICE)
idxs = torch.randint(0, len(dataset), (10,))
imgs = dataset[idxs].flatten(start_dim=1)

with torch.no_grad():
    reconstructed = model(imgs.to(DEVICE))[0]

fig = plt.figure(figsize=(9, 4))
for idx, (img, img_rec) in enumerate(zip(imgs, reconstructed)):
    ax = fig.add_subplot(2, 10, idx+1)
    bx = fig.add_subplot(2, 10, idx+11)
    ax.imshow(img.cpu().reshape(28, 20, 1), cmap="gray")
    bx.imshow(img_rec.cpu().reshape(28, 20, 1), cmap='gray')
plt.show()
