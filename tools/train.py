import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vit import VIT

import torchvision.transforms as transforms
import torchvision
import torch
import sys
import os
import time
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import torch.optim as optim
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import OneCycleLR
import logging


## Create VIT configuration
img_size = 224
C = 3
H = img_size
W = img_size
patch_size = 16
num_patches = H*W // patch_size**2
emb_dim = 768
num_heads = 12
num_layers = 12
hidden_dim = 4*emb_dim
dropout = 0.1
n_classes = 100
img_shape = (H,W)
print(f"Number of channels (C): {C}")
print(f"Height (H): {H}")
print(f"Width (W): {W}")
print(f"Patch size (patch_size): {patch_size}")
print(f"Number of patches (num_patches): {num_patches}")
print(f"Embedding dimension (emb_dim): {emb_dim}")
print(f"Number of heads (num_heads): {num_heads}")
print(f"Head dimension: {emb_dim // num_heads}")
print(f"Number of transformer layers (num_layers): {num_layers}")
print(f"Hidden dimensions: {hidden_dim}")
print(f"Number of classes: {n_classes}")


## Create dataloaders
train_transform = transforms.Compose([
    transforms.Resize(img_size),  # First resize to 224x224
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Keep crop size at 224
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    )
])

val_transform = transforms.Compose([
    transforms.Resize(224),  # Also resize validation images
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    )
])

# cifar100 = torchvision.datasets.CIFAR100(root='../datasets', download=True)
cifar100_train = torchvision.datasets.CIFAR100(
    root='./datasets', 
    train=True,  # Important: Specify train=True
    transform=train_transform,
    download=True
)

cifar100_val = torchvision.datasets.CIFAR100(
    root='./datasets', 
    train=False,  # Using training set for validation
    transform=val_transform,
    download=True
)

# Create DataLoaders for training and validation
batch_size = 128
train_loader = torch.utils.data.DataLoader(
    cifar100_train, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True 
)
val_loader = torch.utils.data.DataLoader(
    cifar100_val, batch_size=batch_size, shuffle=False, num_workers=12
)

dataiter = iter(train_loader)
images, targets = next(dataiter)


vit = VIT(
    num_layers=num_layers,
    num_heads=num_heads,
    emb_dim=emb_dim,
    hidden_dim=hidden_dim,
    patch_size=patch_size,
    C=C,
    img_shape=img_shape,
    n_classes=n_classes,
    dropout=dropout,
)
# Compile the model
vit = torch.compile(vit)

## Calculate number of params
total_params = 0
for name, param in vit.named_parameters():
    if param.requires_grad:
        num_params = param.numel()
        # print(f"{name}: {num_params:,} parameters")
        total_params += num_params
print(f"Total trainable parameters: {total_params:,}")

def train_vit(
    vit,
    train_loader,
    val_loader,
    num_epochs=50,
    max_lr=4e-5,
    max_grad_norm=1.0,
    weight_decay=5e-2,
    seed=42,
    val_freq=1,  # Validate every n epochs
    patience=10,  # Early stopping patience
):
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=1,
        log_with="tensorboard",
        project_dir="training_logs"
    )

    # Setup logging
    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Set seed for reproducibility
    set_seed(seed)
    
    if accelerator.is_main_process:
        logger.info(f"Number of devices: {accelerator.num_processes}")
        logger.info(f"Mixed precision: {accelerator.mixed_precision}")
        logger.info(f"Validation frequency: every {val_freq} epochs")
        logger.info(f"Early stopping patience: {patience} epochs")
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        vit.parameters(),
        lr=max_lr/25,  # base learning rate
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Calculate total steps for scheduler
    total_steps = len(train_loader) * num_epochs
    
    # Initialize scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    # Prepare for distributed training
    vit, optimizer, train_loader, val_loader = accelerator.prepare(
        vit, optimizer, train_loader, val_loader
    )
    
    # Training metrics
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    no_improve = 0  # Counter for early stopping
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        vit.train()
        running_train_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            with accelerator.accumulate(vit):
                optimizer.zero_grad()
                outputs = vit(images)
                loss = F.cross_entropy(outputs, targets)
                
                # Backward pass with accelerator
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    clip_grad_norm_(vit.parameters(), max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                
                running_train_loss += loss.item()
        
        # Calculate average training loss
        train_loss = running_train_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase (run every val_freq epochs)
        val_loss = None
        accuracy = None
        
        if (epoch + 1) % val_freq == 0:
            vit.eval()
            running_val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, targets in val_loader:
                    outputs = vit(images)
                    loss = F.cross_entropy(outputs, targets)
                    running_val_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            val_loss = running_val_loss / len(val_loader)
            accuracy = 100. * correct / total
            val_losses.append(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                # Save best model
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(vit)
                    accelerator.save(
                        unwrapped_model.state_dict(),
                        f"best_model_epoch_{epoch+1}.pth"
                    )
            else:
                no_improve += 1
                if no_improve >= patience:
                    if accelerator.is_main_process:
                        logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Log metrics
        if accelerator.is_main_process:
            epoch_time = time.time() - epoch_start_time
            current_lr = optimizer.param_groups[0]['lr']
            
            log_msg = f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}'
            if val_loss is not None:
                log_msg += f', Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%'
            log_msg += f', LR: {current_lr:.2e}, Time: {epoch_time:.2f}s'
            
            logger.info(log_msg)
            
            # Log to tensorboard
            metrics = {
                "train_loss": train_loss,
                "learning_rate": current_lr,
            }
            if val_loss is not None:
                metrics.update({
                    "val_loss": val_loss,
                    "accuracy": accuracy,
                })
            accelerator.log(metrics, step=epoch)
    
    # Save final model
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(vit)
        accelerator.save(
            unwrapped_model.state_dict(),
            "final_model.pth"
        )
        
    return train_losses, val_losses

# Main execution
if __name__ == "__main__":
    # Call training function
    train_losses, val_losses = train_vit(
        vit=vit,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        max_lr=4e-5,
        max_grad_norm=1.0,
        weight_decay=5e-2,
        val_freq=10,  # Validate every epoch
        patience=10,  # Early stopping patience
    )