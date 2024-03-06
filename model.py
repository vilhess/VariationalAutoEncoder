import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, z_dim):
        super(Encoder, self).__init__()
        self.input_hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden_mu = nn.Linear(hidden_dim, z_dim)
        self.hidden_sigma = nn.Linear(hidden_dim, z_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input_hidden(x))
        mu = self.hidden_mu(x)
        sigma = self.hidden_sigma(x)
        return mu, sigma
    

class Decoder(nn.Module):

    def __init__(self, z_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.z_hidden = nn.Linear(z_dim, hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, z):
        out = self.relu(self.z_hidden(z))
        out = nn.Sigmoid()(self.hidden_out(out))
        return out

class VAE(nn.Module):

    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, z_dim)
        self.decoder = Decoder(z_dim, hidden_dim, input_dim)

    def rep_trick(self, mu, sigma):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma*epsilon
        return z
    
    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self.rep_trick(mu, sigma)
        x_new = self.decoder(z)
        return x_new, mu, sigma
    

if __name__ == '__main__':

    model = VAE(28*28, hidden_dim=200, z_dim=20)
    x = torch.rand((10, 1, 28, 28))

    x_flat = torch.flatten(x, start_dim=1)
    print(x_flat.shape)

    x_new, mu, sigma = model(x_flat)
    print(x_new.shape, mu.shape, sigma.shape)




