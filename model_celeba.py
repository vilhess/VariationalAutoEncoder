import torch
import torch.nn as nn
from torchvision.transforms import functional as tf
from dataset import CelebA

DEVICE='cpu'

class VAE_CelebA(nn.Module):
    def __init__(self, in_channels=3, hiddens_dim=[32, 64, 128, 256, 512], latent_dim=128):
        super(VAE_CelebA, self).__init__()
        in_channels=3
        self.final_dim = hiddens_dim[-1]

        modules = []
        for layer_dim in hiddens_dim:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=layer_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(layer_dim),
                    nn.ReLU()
                )
            )
            in_channels=layer_dim
        self.encoder = nn.Sequential(*modules)

        out = self.encoder(torch.rand(1, 3, 218, 178))
        self.size_a, self.size_b= out.shape[2], out.shape[3]

        self.fc_mu = nn.Linear(hiddens_dim[-1]*self.size_a*self.size_b, latent_dim)
        self.fc_sigma = nn.Linear(hiddens_dim[-1]*self.size_a*self.size_b, latent_dim)

        modules = []
        self.decoder_layer = nn.Linear(latent_dim, hiddens_dim[-1]*self.size_a*self.size_b)
        hiddens_dim.reverse()
        for i in range(len(hiddens_dim)-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hiddens_dim[i], hiddens_dim[i+1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hiddens_dim[i+1]),
                    nn.ReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)
        self.decoder_final_layer = nn.Sequential(
            nn.ConvTranspose2d(hiddens_dim[-1], 3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )


    def encode(self, x):
        result = self.encoder(x)
        result = result.flatten(start_dim=1)
        mu = self.fc_mu(result)
        sigma = self.fc_sigma(result)
        return mu, sigma
    
    def reparameterize(self, mu, sigma):
        epsilon = torch.randn_like(sigma)
        return mu + sigma*epsilon
    
    def decode(self, z):
        out = self.decoder_layer(z)
        out = out.view(-1, self.final_dim, self.size_a, self.size_b)
        out = self.decoder(out)
        out = self.decoder_final_layer(out)
        
        return out
    
    def forward(self, x):
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        out = self.decode(z)
        out = tf.resize(out, x.shape[2:])
        return out, mu, sigma
    

if __name__=='__main__':
    dataset = CelebA('data/CELEBA/images/img_align_celeba')
    model = VAE_CelebA().to(DEVICE)
    img = dataset[0].unsqueeze(0)
    print(img.shape)
    rec_img = model(img.to(DEVICE))[0]
    print(rec_img.shape)