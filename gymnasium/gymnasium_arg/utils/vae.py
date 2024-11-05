import torch
import torch.nn as nn
import torch.optim as optim

class VAE(nn.Module):
    def __init__(self, imu_dim=(50, 10), action_dim=(50, 6), latent_dim=32):
        super(VAE, self).__init__()
        self.imu_dim = imu_dim
        self.action_dim = action_dim

        # IMU Encoding Path
        self.imu_encoder = nn.Sequential(
            nn.Conv1d(imu_dim[1], 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Action Encoding Path
        self.action_encoder = nn.Sequential(
            nn.Conv1d(action_dim[1], 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate encoded dimensions dynamically
        imu_dummy = torch.randn(1, imu_dim[1], imu_dim[0])
        action_dummy = torch.randn(1, action_dim[1], action_dim[0])
        imu_encoded_dim = self.imu_encoder(imu_dummy).shape[1]
        action_encoded_dim = self.action_encoder(action_dummy).shape[1]
        
        # Encoder (Combines IMU and Action Encodings)
        self.encoder_fc = nn.Sequential(
            nn.Linear(imu_encoded_dim + action_encoded_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)  # Mean and log variance
        )
        
        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, imu_dim[0] * imu_dim[1] + action_dim[0] * action_dim[1])
        )
    
    def encode(self, imu, action):
        # Process IMU and Action inputs through separate encoding paths
        imu_encoded = self.imu_encoder(imu)
        action_encoded = self.action_encoder(action)
        
        # Concatenate and pass through fully connected encoder layers
        combined = torch.cat((imu_encoded, action_encoded), dim=-1)
        encoded = self.encoder_fc(combined)
        
        # Split into mean and log variance
        mu, logvar = torch.chunk(encoded, 2, dim=-1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        # Decode combined latent variable
        decoded = self.decoder_fc(z)
        
        # Separate the decoded output for IMU and action
        imu_decoded, action_decoded = torch.split(
            decoded, [self.imu_dim[0] * self.imu_dim[1], self.action_dim[0] * self.action_dim[1]], dim=-1
        )
        
        # Reshape to original 2D array shape for IMU and action
        imu_decoded = imu_decoded.view(-1, self.imu_dim[0], self.imu_dim[1])
        action_decoded = action_decoded.view(-1, self.action_dim[0], self.action_dim[1])
        
        return imu_decoded, action_decoded
    
    def forward(self, imu, action):
        mu, logvar = self.encode(imu, action)
        z = self.reparameterize(mu, logvar)
        imu_recon, action_recon = self.decode(z)
        return imu_recon, action_recon, mu, logvar

# Loss function for VAE
def vae_loss(imu_recon, action_recon, imu, action, mu, logvar, beta=1.0):
    # Reconstruction loss
    recon_loss = nn.functional.mse_loss(imu_recon, imu, reduction='sum') + \
                 nn.functional.mse_loss(action_recon, action, reduction='sum')
    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld

# import torch
# import torch.nn as nn
# import torch.optim as optim

# class VAE(nn.Module):
#     def __init__(self, obs_dim, latent_dim):
#         super(VAE, self).__init__()
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Linear(obs_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, latent_dim * 2)  # Output mean and log variance
#         )
#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 128),
#             nn.ReLU(),
#             nn.Linear(128, obs_dim)
#         )
    
#     def encode(self, x):
#         x = self.encoder(x)
#         mu, logvar = torch.chunk(x, 2, dim=-1)  # Split the output into mean and log variance
#         return mu, logvar
    
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
    
#     def decode(self, z):
#         return self.decoder(z)
    
#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar

# # Loss function for VAE
# def vae_loss(recon_x, x, mu, logvar, beta=1.0):
#     # Reconstruction loss
#     recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
#     # KL divergence
#     kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return recon_loss + beta * kld
