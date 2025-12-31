import torch
import torch.nn.functional as F
from torch import nn
import math
import torch.distributions as td


class VAE(nn.Module):
    # Vanilla Variational Auto-Encoder
    def __init__(self, state_dim, action_dim, latent_dim, beta, device, hidden_dim=750):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim, hidden_dim)
        self.e2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)

        self.d1 = nn.Linear(latent_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.d3 = nn.Linear(hidden_dim, state_dim)

        self.e_1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.e_2 = nn.Linear(hidden_dim, hidden_dim)
        self.e_3 = nn.Linear(hidden_dim, latent_dim)

        self.relu = nn.ReLU()
        self.beta = beta
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state):
        mean, std, log_std = self.encodes(state)
        z = mean + std * torch.randn_like(std)
        return self.decode(z), mean, std, log_std, z

    def R_loss(self, state, action):
        zsa = self.encodesa(state, action)
        next_state = self.decode(zsa)

        mean, std, log_std = self.encodes(next_state)
        z = mean + std * torch.randn_like(std)
        u = self.decode(z)

        recon_loss = ((u - next_state.detach()) ** 2).mean(-1)
        KL_loss = -0.5 * (1 + 2 * log_std - mean.pow(2) - std.pow(2)).mean(-1)
        vae_loss = recon_loss + self.beta * KL_loss
        return vae_loss

    def encodes(self, state):
        z = self.relu(self.e1(state))
        z = self.relu(self.e2(z))
        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        return mean, std, log_std

    def encodesa(self, state, action):
        z = self.relu(self.e_1(torch.cat([state, action], -1)))
        z = self.relu(self.e_2(z))
        return self.e_3(z)

    def decode(self, z):
        s = self.relu(self.d1(z))
        s = self.relu(self.d2(s))
        return self.d3(s)

    def train_vae(self, state, action, next_state):
        recon, mean, std, log_std, z = self.forward(next_state)
        recon_loss = F.mse_loss(recon, next_state)
        KL_loss = -0.5 * (1 + 2 * log_std - mean.pow(2) - std.pow(2)).mean()

        z_sa = self.encodesa(state, action)
        encodesa_loss = F.mse_loss(z.detach(), z_sa)  # F.mse_loss(mean, z_sa)

        vae_loss = recon_loss + self.beta * KL_loss + encodesa_loss
        return vae_loss

