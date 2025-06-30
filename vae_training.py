import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os

# Define Variational Autoencoder for design parameters
class DesignVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=4):
        super(DesignVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl_div

def train_vae(X, epochs=100, batch_size=32, latent_dim=4, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    input_dim = X.shape[1]

    dataset = TensorDataset(torch.tensor(X.values, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DesignVAE(input_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(device)
            recon_x, mu, logvar = model(x)
            loss = vae_loss(recon_x, x, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch} Loss: {total_loss:.4f}")
    return model

def generate_new_designs(model, num_samples=10, latent_dim=4, feature_names=None, output_path="generated_designs.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.eval()

    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        generated = model.decode(z).cpu().numpy()

    df_generated = pd.DataFrame(generated, columns=feature_names if feature_names else [f"Feature_{i}" for i in range(generated.shape[1])])
    df_generated.to_csv(output_path, index=False)
    print(f"✅ {num_samples} new designs saved to {output_path}")
    return df_generated
