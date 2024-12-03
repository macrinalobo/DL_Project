import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 256), 
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.fc3_mean = nn.Linear(256, latent_dim)  
        self.fc3_logvar = nn.Linear(256, latent_dim) 
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        mean = self.fc3_mean(x)
        logvar = self.fc3_logvar(x)
        
        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(latent_dim, 256),  
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 1024), 
            nn.LayerNorm(1024),
            nn.ReLU(),
        )
        self.fc3 = nn.Linear(1024, output_dim)
        self.dropout = nn.Dropout(0.8)

    def forward(self, z):
        z = self.fc1(z)
        z = self.dropout(z)
        z = self.fc2(z)
        z = self.dropout(z)
        output = self.fc3(z)
        return output

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z
    
class VAE2(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(VAE2, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, output_dim)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z
    
def compute_neighborhood_loss(latent_means_all, closest_neighbors, furthest_neighbors):
    """
    Compute the neighborhood loss after the entire epoch, using the latent representations
    of all data points and their spatial neighbors.
    """
    neighborhood_loss = 0
    for i, _ in closest_neighbors.items():
        latent_point = latent_means_all[i] 
        close_latent_neighbors = [latent_means_all[neighbor] for neighbor in closest_neighbors[i]]
        far_latent_neighbors = [latent_means_all[neighbor] for neighbor in furthest_neighbors[i]]
        
        dist_to_closest_neighbors = torch.stack([torch.norm(latent_point - neighbor, dim=0) for neighbor in close_latent_neighbors])
        dist_to_furthest_neighbors = torch.stack([torch.norm(latent_point - neighbor, dim=0) for neighbor in far_latent_neighbors]) 
        
        neighborhood_loss += dist_to_closest_neighbors.sum() - dist_to_furthest_neighbors.sum()

    return neighborhood_loss

def vae_loss(reconstructed_x, x, mean, logvar, lambda_kl=1e-2):
    MSE = F.mse_loss(reconstructed_x, x, reduction='sum')  
    KL_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return MSE + lambda_kl * KL_div 


def vae_loss2(reconstructed_x, x, mean, logvar, closest_neighbors, furthest_neighbors, lambda_kl=1e-2, lambda_nl=1e-2):
    MSE = F.mse_loss(reconstructed_x, x, reduction='sum')  
    KL_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    neigh_loss = compute_neighborhood_loss(mean, closest_neighbors, furthest_neighbors)
    return MSE + lambda_kl * KL_div + lambda_nl * neigh_loss
    
def vae_loss3(reconstructed_x, x, mean, logvar, lambda_kl=1e-2, lambda_nl=1e-2):
    """
    utilizes KL divergence and poisson NLL loss
    """
    # print(reconstructed_x.shape)
    # print(x.shape)
    # print(logvar.shape)
    MSE = F.mse_loss(reconstructed_x, x, reduction='sum')
    KL_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    loss_p_nll = nn.PoissonNLLLoss()
    loss_pnll = loss_p_nll(reconstructed_x, x)
    return MSE + lambda_kl * KL_div + lambda_nl * loss_pnll

def plot_umap(data, title, save_path):
    """Helper function to plot UMAP"""
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_embedding = reducer.fit_transform(data)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], s=10)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend()
    plt.savefig(save_path)