import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from PIL import Image

from model_celeba import VAE_CelebA
from dataset import CelebA

space_rep = Image.open('pics/2dMNIST.png')

DEVICE = 'mps'

transform = transforms.ToPILImage()

dataset = st.sidebar.selectbox('Dataset', ['MNIST', 'Frey Face', 'CelebA'])
celeba_dataset = CelebA('data/CELEBA/images/img_align_celeba')

def plot(model):
    idxs = torch.randint(0, len(celeba_dataset), (5,))
    imgs = []
    for idx in idxs:
        imgs.append(celeba_dataset[idx])
    imgs = torch.stack(imgs)

    with torch.no_grad():
        reconstructed = model(imgs.to('cpu'))[0]

    fig = plt.figure(figsize=(9, 4))
    for idx, (img, img_rec) in enumerate(zip(imgs, reconstructed)):
        ax = fig.add_subplot(2, 5, idx+1)
        bx = fig.add_subplot(2, 5, idx+6)
        ax.imshow(img.cpu().permute(1, 2, 0), cmap="gray")
        bx.imshow(img_rec.cpu().permute(1, 2, 0), cmap='gray')
        ax.axis('off')
        bx.axis('off')
    st.pyplot(fig)

if 'loaded' not in st.session_state :

    st.session_state['mnist'] = torch.load('models/mnist-2dv2.pkl').to(DEVICE)
    st.session_state['ff'] = torch.load('models/ff-8d.pkl').to(DEVICE)
    checkpoints = torch.load('models/checkpoint-29.pth', map_location='cpu')
    parameters = checkpoints['model_state_dict']
    model_celeba = VAE_CelebA()
    model_celeba.load_state_dict(parameters)
    st.session_state['celeba'] = model_celeba
    st.session_state['loaded'] = True



if dataset=='MNIST':
    st.title('MNIST')
    model = st.session_state['mnist']
elif dataset=='Frey Face':
    st.title('Frey face')
    model = st.session_state['ff']
else:
    st.title('CelebA')
    model = st.session_state['celeba']
    decoder = model.decode

if dataset!='CelebA':
    model.eval()
    decoder = model.decoder

if dataset=='MNIST':
    st.image(space_rep, caption="Latent space")
if dataset!='CelebA':

    col1, col2 = st.columns(2)
    with col1:

        coord1 = st.slider("coord1", float(-6), float(6), float(0), step=0.1)
        coord2 = st.slider("coord2", float(-6), float(6), float(0), step=0.1)

        if dataset=='Frey Face':

            coord3 = st.slider("coord3", float(-6), float(6), float(0), step=0.1)
            coord4 = st.slider("coord4 (smile)", float(-6), float(6), float(0), step=0.1)
            coord5 = st.slider("coord5", float(-6), float(6), float(0), step=0.1)
            coord6 = st.slider("coord6 (face orientation)", float(-6), float(6), float(0), step=0.1)
            coord7 = st.slider("coord7", float(-6), float(6), float(0), step=0.1)
            coord8 = st.slider("coord8", float(-6), float(6), float(0), step=0.1)

    if dataset=='MNIST':
        output = decoder(torch.Tensor((coord1, coord2)).to(DEVICE)).cpu().detach().reshape(1, 28, 28)
    else:
        output = decoder(torch.Tensor((coord1, coord2, coord3, coord4, coord5, coord6, coord7, coord8)).to(DEVICE)).cpu().detach().reshape(1, 28, 20)
    img = transform(output)

    with col2:
        st.image(img, use_column_width=True, caption="digit generated from coordinates")

else:
    st.subheader('The latent space dimension has a dimension of 128')
    if st.button('reload'):
        plot(model)