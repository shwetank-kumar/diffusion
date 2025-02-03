#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

# ## Check for CUDA

# In[2]:


# Check if GPU is available
if torch.cuda.is_available():
    print("GPU is available!")
else:
    print("GPU is not available. Using CPU.")
    
# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ## Download data

# In[3]:


celeba = torchvision.datasets.CelebA(root='./', download=True)

# In[4]:


transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to a consistent size
    transforms.ToTensor(),  # Convert PIL Images to tensors
])

# Assuming celeba is your dataset, apply the transform
celeba.transform = transform

# In[5]:


celeb_loader = torch.utils.data.DataLoader(celeba,
                                          batch_size=16,
                                          shuffle=True,
                                          num_workers=12)

# ## Plot a batch

# In[6]:


import matplotlib.pyplot as plt
def show_celeb_batch(images):
    """
    Display a batch of CelebA images with their corresponding names
    
    Args:
        images (torch.Tensor): Batch of images (B x C x H x W)
        names (list): List of names corresponding to the images
    """
    # Convert images from tensor to numpy array
    images = images.numpy()
    
    # Move channel dimension to end for plotting (B x H x W x C)
    images = np.transpose(images, (0, 2, 3, 1))
    
    # Get number of images in batch
    batch_size = images.shape[0]
    
    # Create figure
    fig, axes = plt.subplots(1, batch_size, figsize=(12, 3))
    if batch_size == 1:
        axes = [axes]
    
    # Plot each image with its name
    for idx, (img, ax) in enumerate(zip(images, axes)):
        ax.imshow(img)
        ax.axis('off')
    
    plt.tight_layout()
    return fig

# In[7]:


# Get a batch of images and their corresponding names
dataiter = iter(celeb_loader)
images, _ = next(dataiter)
images = images.to(device)

# Show the batch
fig = show_celeb_batch(images.cpu())
plt.show()


# In[8]:


print("Image shape: ",images.shape)
print(f"Max pixel value: {images.max()}, Min pixel value: {images.min()}")

# ## Create VAE architecture

# In[9]:


import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    """Initialize weights for convolutional and linear layers."""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            # He initialization for convolutional layers with ReLU/LeakyReLU
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        elif isinstance(m, nn.Linear):
            # He initialization for linear layers with ReLU/LeakyReLU
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)  # Initialize biases to 0

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Layer 1: 128x128x3 → 64x64x64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.01),
        )
        # Layer 2: 64x64x64 → 32x32x128
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.01),
        )
        # Layer 3: 32x32x128 → 16x16x256
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.01),
        )
        # Layer 4: 16x16x256 → 8x8x512
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.01),
        )
        # Layer 5: 8x8x512 → 4x4x1024
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.mean_lin = nn.Linear(in_features=1024 * 4 * 4, out_features=1024)  # Latent dim = 1024
        self.logvar_lin = nn.Linear(in_features=1024 * 4 * 4, out_features=1024)  # Latent dim = 1024
        
        # Initialize weights
        self.apply(init_weights)
        
    def forward(self, x):
        # Save feature maps for skip connections
        x1 = self.conv1(x)  # 64x64x64
        x2 = self.conv2(x1)  # 32x32x128
        x3 = self.conv3(x2)  # 16x16x256
        x4 = self.conv4(x3)  # 8x8x512
        x5 = self.conv5(x4)  # 4x4x1024
        
        x5 = torch.flatten(x5, start_dim=1)  # Flatten to (batch_size, 1024 * 4 * 4)
        mean = self.mean_lin(x5)
        logvar = self.logvar_lin(x5)
        return mean, logvar, [x1, x2, x3, x4]  # Return feature maps for skip connections

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.input = nn.Linear(1024, 1024 * 4 * 4)  # Match latent_dim=1024
        
        # Layer 1: 4x4x1024 → 8x8x1024
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(),
        )
        # Layer 2: 8x8x1024 + 8x8x512 → 16x16x512
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024 + 512, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
        )
        # Layer 3: 16x16x512 + 16x16x256 → 32x32x256
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512 + 256, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
        )
        # Layer 4: 32x32x256 + 32x32x128 → 64x64x128
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256 + 128, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
        )
        # Layer 5: 64x64x128 + 64x64x64 → 128x128x64
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128 + 64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
        )
        # Layer 6: 128x128x64 → 128x128x3
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),  # Output in range [0, 1]
        )
        
        # Initialize weights
        self.apply(init_weights)
    
    def forward(self, x, skip_connections):
        x = self.input(x)
        x = x.view(-1, 1024, 4, 4)  # Reshape to (batch_size, 1024, 4, 4)
        
        # Decoder with skip connections
        x = self.conv1(x)  # 8x8x1024
        x = torch.cat([x, skip_connections[3]], dim=1)  # Concatenate with encoder feature map (8x8x512)
        x = self.conv2(x)  # 16x16x512
        x = torch.cat([x, skip_connections[2]], dim=1)  # Concatenate with encoder feature map (16x16x256)
        x = self.conv3(x)  # 32x32x256
        x = torch.cat([x, skip_connections[1]], dim=1)  # Concatenate with encoder feature map (32x32x128)
        x = self.conv4(x)  # 64x64x128
        x = torch.cat([x, skip_connections[0]], dim=1)  # Concatenate with encoder feature map (64x64x64)
        x = self.conv5(x)  # 128x128x64
        x = self.conv6(x)  # 128x128x3
        return x

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.enc = Encoder()
        self.dec = Decoder()
        
    def forward(self, x):
        mean, logvar, skip_connections = self.enc(x)  # Get skip connections from encoder
        std = torch.exp(0.5 * logvar) + 1e-8  # Numerical stability
        eps = torch.randn_like(std)
        latent_samples = mean + eps * std
        out = self.dec(latent_samples, skip_connections)  # Pass skip connections to decoder
        return mean, logvar, out

# In[10]:


vae = VAE().to(device)

# In[11]:


mean, std, out = vae(images)

# In[12]:


mean.shape, std.shape, out.shape
out = out.detach()

# ## Plot out of untrained VAE

# In[13]:


# Show the vae output batch
fig = show_celeb_batch(out.cpu())
plt.show()
print("Image shape: ",out.shape)
print(f"Max pixel value: {out.max()}, Min pixel value: {out.min()}")

# ## Create loss function

# In[14]:


## 2 parts to loss BCE and KL divergence
def loss(target_image, predicted_image, mean, logvar, beta):
    bce_loss = F.binary_cross_entropy(predicted_image, target_image, reduction='mean')
    kldiv_loss = -0.5*torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    # print(f"BCE Loss: {bce_loss.item()}, KL Divergence Loss: {kldiv_loss.item()}")
    total_loss = bce_loss + beta*kldiv_loss
    return bce_loss, kldiv_loss, total_loss

# ## Training loop

# In[15]:


import matplotlib.pyplot as plt

def visualize_reconstructions(target_images, reconstructed_images, num_images=4):
    fig, axes = plt.subplots(2, num_images, figsize=(15, 6))
    for i in range(num_images):
        # Original image
        axes[0, i].imshow(target_images[i].permute(1, 2, 0).cpu().numpy())
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')

        # Reconstructed image
        axes[1, i].imshow(reconstructed_images[i].permute(1, 2, 0).cpu().numpy())
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis('off')
    plt.show()

# In[ ]:


import torch.optim as optim
num_epochs = 10
beta = 0.01
optimizer = optim.Adam(vae.parameters(), lr=1e-4)  # Increased learning rate

# Training loop
for epoch in range(num_epochs):
    vae.train()  # Set model to training mode
    epoch_bce_loss = 0.0
    epoch_kldiv_loss = 0.0
    epoch_total_loss = 0.0
    
    # Iterate over all batches in the dataset
    for batch_idx, (images, _) in enumerate(celeb_loader):
        # Move images to the appropriate device
        images = images.to(device)
        
        # Forward pass
        mean, logvar, predicted_images = vae(images)
        
        # Compute loss
        bce_loss, kldiv_loss, total_loss = loss(images, predicted_images, mean, logvar, beta=beta)
        
        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Print every N batches
        if batch_idx % 1000 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(celeb_loader)}], "
                  f"BCE: {bce_loss:.4f}, KL: {kldiv_loss:.4f}, Loss: {total_loss:.4f}")
        
        # Accumulate losses for logging
        epoch_bce_loss += bce_loss.item()
        epoch_kldiv_loss += kldiv_loss.item()
        epoch_total_loss += total_loss.item()
    
    # Print average losses for the epoch
    num_batches = len(celeb_loader)
    avg_bce_loss = epoch_bce_loss / num_batches
    avg_kldiv_loss = epoch_kldiv_loss / num_batches
    avg_total_loss = epoch_total_loss / num_batches
    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Avg BCE Loss: {avg_bce_loss:.4f}, "
          f"Avg KL Div Loss: {avg_kldiv_loss:.4f}, "
          f"Avg Loss: {avg_total_loss:.4f}")
    
    # Visualize reconstructions periodically (e.g., every 10 epochs)
    if (epoch + 1) % 1 == 0:
        vae.eval()  # Set model to evaluation mode
        with torch.no_grad():
            # Use a fixed batch for visualization (e.g., the first batch)
            fixed_images, _ = next(iter(celeb_loader))
            fixed_images = fixed_images.to(device)
            _, _, reconstructed_images = vae(fixed_images)
            visualize_reconstructions(fixed_images, reconstructed_images)
        vae.train()  # Set model back to training mode

# ## Save model

# In[ ]:


# Save model
torch.save(vae.state_dict(), 'vae_weights.pth')
