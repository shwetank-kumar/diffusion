#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ## Load model

# In[2]:


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

# In[3]:


vae_test = VAE()
vae_test.load_state_dict(torch.load('vae_weights.pth'))
vae_test.to(device)
vae_test.eval()

# ## Load data

# In[4]:


import torchvision
import torchvision.transforms as transforms

celeba = torchvision.datasets.CelebA(root='./', download=True)

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to a consistent size
    transforms.ToTensor(),  # Convert PIL Images to tensors
])

# Assuming celeba is your dataset, apply the transform
celeba.transform = transform

celeb_loader = torch.utils.data.DataLoader(celeba,
                                          batch_size=16,
                                          shuffle=True,
                                          num_workers=12)

# ## Latent cluster analysis

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm  # For progress bar
import torch

def analyze_latent_clusters(model, dataloader, attribute_index=0):
    """
    Analyze latent clusters of a trained autoencoder model.

    Args:
        model: Trained autoencoder model.
        dataloader: DataLoader providing images and attributes.
        attribute_index: Index of the attribute to use for coloring the clusters.

    Returns:
        latent_2d: 2D latent space representations after t-SNE.
        attributes: Corresponding attributes for visualization.
    """
    latent_vectors = []
    attributes = []
    
    # Set model to evaluation mode
    model.eval()
    
    # Determine the device (CPU or GPU)
    device = next(model.parameters()).device
    
    # Disable gradient computation for efficiency
    with torch.no_grad():
        # Add a progress bar for better feedback
        for images, attrs in tqdm(dataloader, desc="Encoding images"):
            # Move images and attributes to the same device as the model
            images = images.to(device)
            
            # Get latent representations
            mu, _, _ = model.enc(images)  # Assuming model.enc returns mu, logvar, and other outputs
            latent_vectors.append(mu.cpu().numpy())  # Move back to CPU for NumPy operations
            attributes.append(attrs.cpu().numpy())  # Move back to CPU for NumPy operations
    
    # Concatenate all latent vectors and attributes
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    attributes = np.concatenate(attributes, axis=0)
    
    # Reduce dimensionality using t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_vectors)
    
    # Plot clusters colored by the selected attribute
    plt.figure(figsize=(10, 8))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=attributes[:, attribute_index], cmap='viridis', alpha=0.6)
    plt.colorbar(label=f'Attribute {attribute_index}')
    plt.title('t-SNE Visualization of Latent Space Colored by Attribute')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.show()
    
    return latent_2d, attributes

# In[ ]:


latent_2d, attributes = analyze_latent_clusters(vae_test, celeb_loader, attribute_index=20)  # Attribute 20 might correspond to "Smiling"

# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np

def plot_latent_attributes(latent_vectors, attributes, attribute_names, figsize=(20, 20)):
    # First, compute t-SNE embedding if not already done
    # We only need to do this once for all attributes
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_vectors)
    
    # Calculate number of rows/columns for subplot grid
    n_plots = len(attribute_names)
    n_cols = 3  # We'll do 3 columns
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # Create subplot for each attribute
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.ravel()  # Flatten axes array for easier indexing
    
    for idx, (attr_name, attr_values) in enumerate(zip(attribute_names, attributes.T)):
        ax = axes[idx]
        
        # Create scatter plot
        scatter = ax.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                           c=attr_values, cmap='viridis',
                           alpha=0.5, s=10)
        
        # Add title and colorbar
        ax.set_title(f'Attribute: {attr_name}')
        plt.colorbar(scatter, ax=ax)
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
    
    # Remove any empty subplots
    for idx in range(len(attribute_names), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    return fig

# Example usage:
def analyze_all_attributes(model, dataloader):
    # Collect latent vectors and attributes
    latent_vectors = []
    all_attributes = []
    
    with torch.no_grad():
        for images, attrs in dataloader:
            images = images.to(device)
            # Get latent representations (using just the mean)
            mu, _, _ = model.enc(images)
            latent_vectors.append(mu.cpu().numpy())
            all_attributes.append(attrs.cpu().numpy())
    
    latent_vectors = np.concatenate(latent_vectors)
    all_attributes = np.concatenate(all_attributes)
    
    # Get your attribute names (replace with actual names from your dataset)
    attribute_names = [
        "Smiling", "Eyeglasses", "Male", "Young", 
        "Wavy_Hair", "Blond_Hair", "Heavy_Makeup"
    ]  # Add all attributes you want to analyze
    
    # Create visualization
    fig = plot_latent_attributes(latent_vectors, all_attributes, attribute_names)
    plt.show()

# In[ ]:


analyze_all_attributes(vae_test, celeb_loader)

# ## Interpolation analysis

# In[14]:


# def interpolate_latent_points(model, start_img, end_img, steps=10):
#     model.eval()
    
#     if start_img.dim() == 3:
#         start_img = start_img.unsqueeze(0)
#     if end_img.dim() == 3:
#         end_img = end_img.unsqueeze(0)
    
#     start_img = start_img.cuda()
#     end_img = end_img.cuda()
    
#     with torch.no_grad():
#         # Get latent representations and skip connections for both images
#         start_z, _, start_skips = model.enc(start_img)
#         end_z, _, end_skips = model.enc(end_img)
        
#         alphas = np.linspace(0, 1, steps)
#         interp_images = []
        
#         for alpha in alphas:
#             # Interpolate in latent space
#             z = start_z * (1-alpha) + end_z * alpha
            
#             # Interpolate skip connections
#             interpolated_skips = []
#             for start_skip, end_skip in zip(start_skips, end_skips):
#                 interpolated_skip = start_skip * (1-alpha) + end_skip * alpha
#                 interpolated_skips.append(interpolated_skip)
            
#             # Decode with interpolated skip connections
#             decoded_img = model.dec(z, interpolated_skips)
#             interp_images.append(decoded_img)
        
#         # Stack images for visualization
#         images = torch.stack(interp_images)
        
#         # Calculate grid dimensions
#         total_images = steps + 2  # Include start and end images
#         n_cols = min(8, total_images)  # Max 8 images per row
#         n_rows = (total_images + n_cols - 1) // n_cols
        
#         # Create the plot
#         plt.figure(figsize=(20, 4 * n_rows))
        
#         # Plot original start image
#         plt.subplot(n_rows, n_cols, 1)
#         plt.imshow(start_img.cpu().squeeze().permute(1, 2, 0).clip(0, 1))
#         plt.axis('off')
#         plt.title('Start')
        
#         # Plot interpolated images
#         for i in range(steps):
#             plt.subplot(n_rows, n_cols, i+2)
#             img = images[i].cpu().squeeze().permute(1, 2, 0).clip(0, 1)
#             plt.imshow(img)
#             plt.axis('off')
#             plt.title(f'α={alphas[i]:.2f}')
        
#         # Plot original end image
#         plt.subplot(n_rows, n_cols, steps+2)
#         plt.imshow(end_img.cpu().squeeze().permute(1, 2, 0).clip(0, 1))
#         plt.axis('off')
#         plt.title('End')
        
#         plt.tight_layout()
#         plt.show()
        
#         return interp_images
%matplotlib notebook 
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt

def interpolate_latent_points(model, start_img, end_img, steps=10, save_movie=True):
    model.eval()
    
    if start_img.dim() == 3:
        start_img = start_img.unsqueeze(0)
    if end_img.dim() == 3:
        end_img = end_img.unsqueeze(0)
    
    start_img = start_img.cuda()
    end_img = end_img.cuda()
    
    with torch.no_grad():
        # Get latent representations and skip connections for both images
        start_z, _, start_skips = model.enc(start_img)
        end_z, _, end_skips = model.enc(end_img)
        
        alphas = np.linspace(0, 1, steps)
        interp_images = []
        
        for alpha in alphas:
            # Interpolate in latent space
            z = start_z * (1-alpha) + end_z * alpha
            
            # Interpolate skip connections
            interpolated_skips = []
            for start_skip, end_skip in zip(start_skips, end_skips):
                interpolated_skip = start_skip * (1-alpha) + end_skip * alpha
                interpolated_skips.append(interpolated_skip)
            
            # Decode with interpolated skip connections
            decoded_img = model.dec(z, interpolated_skips)
            interp_images.append(decoded_img)
        
        # Stack images and prepare for animation
        images = torch.stack(interp_images)
        
        # Create animation
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.close()  # Prevent display of empty figure
        
        def init():
            ax.clear()
            return []
        
        def animate(i):
            ax.clear()
            if i == 0:
                img = start_img.cpu().squeeze().permute(1, 2, 0).clip(0, 1)
                title = 'Start'
            elif i == len(alphas) + 1:
                img = end_img.cpu().squeeze().permute(1, 2, 0).clip(0, 1)
                title = 'End'
            else:
                img = images[i-1].cpu().squeeze().permute(1, 2, 0).clip(0, 1)
                title = f'α={alphas[i-1]:.2f}'
            
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(title)
            return []
        
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                     frames=len(alphas) + 2,  # +2 for start and end frames
                                     interval=200,  # 200ms between frames
                                     blit=True)
        
        if save_movie:
            # Save as MP4
            writer = animation.FFMpegWriter(fps=5, bitrate=2000)
            anim.save('interpolation.mp4', writer=writer)
        else:
            plt.show()
        
        return interp_images, anim

# In[16]:


from IPython.display import HTML
dataiter = iter(celeb_loader)
images, _ = next(dataiter)
# interp_imgs = interpolate_latent_points(vae_test, images[0].unsqueeze(0), images[1].unsqueeze(0), steps=25)
images, anim = interpolate_latent_points(vae_test, images[2].unsqueeze(0), images[3].unsqueeze(0), save_movie=False)
HTML(anim.to_jshtml())

# In[54]:


import math
def curved_interpolate_latent_points(model, start_img, end_img, steps=24, curve_amount=0.5):
   model.eval()
   
   if start_img.dim() == 3:
       start_img = start_img.unsqueeze(0)
   if end_img.dim() == 3:
       end_img = end_img.unsqueeze(0)
   
   start_img = start_img.cuda()
   end_img = end_img.cuda()
   
   with torch.no_grad():
       # Get latent representations and skip connections
       start_z, _, start_skips = model.enc(start_img)
       end_z, _, end_skips = model.enc(end_img)
       
       alphas = np.linspace(0, 1, steps)
       interp_images = []
       
       # Create perpendicular vector for curved path
       mid_point = (start_z + end_z) / 2
       perpendicular = torch.randn_like(start_z).cuda()
       path_direction = end_z - start_z
       # Make perpendicular by subtracting projection
       perpendicular = perpendicular - torch.sum(perpendicular * path_direction) * path_direction / torch.sum(path_direction * path_direction)
       perpendicular = perpendicular / torch.norm(perpendicular)
       
       for alpha in alphas:
           # Create curved path interpolation
           t = alpha * math.pi
           z = (1-alpha) * start_z + alpha * end_z + \
               curve_amount * math.sin(t) * perpendicular
           
           # Interpolate skip connections
           interpolated_skips = []
           for start_skip, end_skip in zip(start_skips, end_skips):
               interpolated_skip = start_skip * (1-alpha) + end_skip * alpha
               interpolated_skips.append(interpolated_skip)
           
           # Decode with interpolated skip connections
           decoded_img = model.dec(z, interpolated_skips)
           interp_images.append(decoded_img)
       
       # Stack images for visualization
       images = torch.stack(interp_images)
       
       # Calculate grid dimensions
       total_images = steps + 2  # Include start and end images
       n_cols = min(8, total_images)  # Max 8 images per row
       n_rows = (total_images + n_cols - 1) // n_cols
       
       # Create the plot
       plt.figure(figsize=(20, 4 * n_rows))
       
       # Plot original start image
       plt.subplot(n_rows, n_cols, 1)
       plt.imshow(start_img.cpu().squeeze().permute(1, 2, 0).clip(0, 1))
       plt.axis('off')
       plt.title('Start')
       
       # Plot interpolated images
       for i in range(steps):
           plt.subplot(n_rows, n_cols, i+2)
           img = images[i].cpu().squeeze().permute(1, 2, 0).clip(0, 1)
           plt.imshow(img)
           plt.axis('off')
           plt.title(f'α={alphas[i]:.2f}')
       
       # Plot original end image
       plt.subplot(n_rows, n_cols, steps+2)
       plt.imshow(end_img.cpu().squeeze().permute(1, 2, 0).clip(0, 1))
       plt.axis('off')
       plt.title('End')
       
       plt.tight_layout()
       plt.show()
       
       return interp_images

# Usage:
# interpolate_latent_points(model, start_img, end_img, steps=24, curve_amount=0.5)

# In[ ]:


interp_imgs = curved_interpolate_latent_points(vae_test, images[3].unsqueeze(0), images[5].unsqueeze(0), steps=25)

# ## Find latent neighbors

# In[11]:


import matplotlib.pyplot as plt
def find_latent_neighbors(model, query_img, dataloader, k=5):
    model.eval()
    
    if query_img.dim() == 3:
        query_img = query_img.unsqueeze(0)
    query_img = query_img.cuda()
    
    with torch.no_grad():
        # Get query latent vector
        query_z, _, _ = model.enc(query_img)
        
        # Process dataset in chunks and keep track of best matches
        best_distances = torch.ones(k, device='cuda') * float('inf')
        best_indices = torch.zeros(k, dtype=torch.long, device='cuda')
        best_images = torch.zeros(k, *query_img.shape[1:], device='cuda')
        current_idx = 0
        
        for batch_imgs, _ in dataloader:
            batch_size = batch_imgs.shape[0]
            batch_imgs = batch_imgs.cuda()
            
            # Get batch encodings
            batch_z, _, _ = model.enc(batch_imgs)
            
            # Calculate distances for this batch
            distances = torch.cdist(query_z, batch_z)
            distances = distances.squeeze()
            
            # Update best matches
            for i in range(batch_size):
                dist = distances[i]
                if dist < best_distances.max():
                    # Find position to insert
                    insert_idx = torch.searchsorted(best_distances.cpu(), dist.cpu()).item()
                    
                    # Create temporary copies for the shift
                    temp_distances = best_distances.clone()
                    temp_indices = best_indices.clone()
                    temp_images = best_images.clone()
                    
                    # Shift everything down
                    if insert_idx < k-1:  # Only shift if not inserting at the end
                        best_distances[insert_idx+1:] = temp_distances[insert_idx:-1]
                        best_indices[insert_idx+1:] = temp_indices[insert_idx:-1]
                        best_images[insert_idx+1:] = temp_images[insert_idx:-1]
                    
                    # Insert new values
                    best_distances[insert_idx] = dist
                    best_indices[insert_idx] = current_idx + i
                    best_images[insert_idx] = batch_imgs[i]
            
            current_idx += batch_size
            
            # Clear GPU memory
            del batch_z, distances
            torch.cuda.empty_cache()
        
        # Move to CPU for visualization
        best_distances = best_distances.cpu()
        best_indices = best_indices.cpu()
        best_images = best_images.cpu()
        
        # Visualize results
        plt.figure(figsize=(2*k, 4))
        
        # Plot query image
        plt.subplot(2, k, k//2 + 1)
        plt.imshow(query_img.cpu().squeeze().permute(1, 2, 0).clip(0, 1))
        plt.axis('off')
        plt.title('Query Image')
        
        # Plot nearest neighbors
        for i in range(k):
            plt.subplot(2, k, k + i + 1)
            plt.imshow(best_images[i].permute(1, 2, 0).clip(0, 1))
            plt.axis('off')
            plt.title(f'Neighbor {i+1}\nDist: {best_distances[i]:.2f}')
        
        plt.tight_layout()
        plt.show()
        
        return best_indices, best_distances

# In[ ]:


dataiter = iter(celeb_loader)
images, _ = next(dataiter)
_, _ = find_latent_neighbors(vae_test, images[5], celeb_loader, k=5)

# In[ ]:



