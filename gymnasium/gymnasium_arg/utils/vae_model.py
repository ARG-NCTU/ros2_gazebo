import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
latent_dim = 20
batch_size = 128
learning_rate = 1e-3
num_epochs = 10

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Encoder network
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(28*28, 400)
        self.relu = nn.ReLU()
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)

    def forward(self, x):
        h = self.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# Decoder network
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 400)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(400, 28*28)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h = self.relu(self.fc1(z))
        x_reconst = self.sigmoid(self.fc2(h))
        return x_reconst

# VAE model combining Encoder and Decoder
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Sample from standard normal
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_reconst = self.decoder(z)
        return x_reconst, mu, logvar

# Loss function: Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(x_reconst, x, mu, logvar):
    # Reconstruction loss
    BCE = nn.functional.binary_cross_entropy(x_reconst, x, reduction='sum')
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Initialize model, optimizer
model = VAE(latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize TensorBoard writer
writer = SummaryWriter('runs/vae_experiment')

# Training loop
model.train()
global_step = 0  # Global step for TensorBoard

for epoch in range(num_epochs):
    train_loss = 0
    for idx, (images, _) in enumerate(train_loader):
        images = images.view(-1, 28*28).to(device)
        optimizer.zero_grad()
        x_reconst, mu, logvar = model(images)
        loss = loss_function(x_reconst, images, mu, logvar)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        global_step += 1

        # Log loss per batch to TensorBoard
        writer.add_scalar('Loss/Batch', loss.item(), global_step)

    average_loss = train_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')

    # Log average loss per epoch to TensorBoard
    writer.add_scalar('Loss/Epoch', average_loss, epoch + 1)

    # Optionally, log reconstructed images to TensorBoard
    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            z = torch.randn(64, latent_dim).to(device)
            sample = model.decoder(z).cpu()
            sample = sample.view(-1, 1, 28, 28)
            grid = torchvision.utils.make_grid(sample)
            writer.add_image('Reconstructed Images', grid, epoch + 1)

# Close the TensorBoard writer
writer.close()

# Save the trained model
torch.save(model.state_dict(), 'vae.pth')
