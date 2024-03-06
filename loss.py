import torch
import torch.nn as nn

class Loss_VAE(nn.Module):

    def __init__(self):
        super().__init__()
        self.rec_loss = nn.BCELoss(reduction="sum")
        
    def kl_div(self, mu, sigma):
        return -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
    
    def forward(self, x, x_new, mu, sigma):
        rec_loss = self.rec_loss(x_new, x)
        kl = self.kl_div(mu, sigma)
        return rec_loss + kl

