import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vit import VIT

import torchvision.transforms as transforms
import torchvision
import torch
import time
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import torch.optim as optim
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import OneCycleLR
import logging
from pathlib import Path
import yaml

def save_checkpoint(
    accelerator, 
    vit, 
    optimizer, 
    scheduler, 
    epoch, 
    train_losses, 
    val_losses,
    best_val_loss,
    no_improve,
    checkpoint_dir
):
    """Save training checkpoint"""
    if accelerator.is_main_process:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        unwrapped_model = accelerator.unwrap_model(vit)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'no_improve': no_improve
        }
        
        accelerator.save(checkpoint, checkpoint_path)
        
        # Save latest checkpoint reference
        with open(os.path.join(checkpoint_dir, "latest_checkpoint.txt"), "w") as f:
            f.write(checkpoint_path)

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory allocated: {allocated:.2f}GB")
        print(f"GPU Memory reserved: {reserved:.2f}GB")

def load_checkpoint(checkpoint_path, vit, optimizer, scheduler):
    """Load training checkpoint"""
    if not os.path.exists(checkpoint_path):
        return None
        
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    vit.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:  # Only load scheduler state if scheduler is provided
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint

def get_latest_checkpoint(checkpoint_dir):
    """Get the path of the latest checkpoint"""
    latest_file = os.path.join(checkpoint_dir, "latest_checkpoint.txt")
    if os.path.exists(latest_file):
        with open(latest_file, "r") as f:
            return f.read().strip()
    return None

def check_gpu_temperature(accelerator, max_temp=80):
    """Check GPU temperature for the current process's GPU."""
    print(f" Starting temperature check.")
    try:
        # Get GPU temperatures using nvidia-smi
        temp = os.popen('nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader').read()
        if temp:
            # Split temperatures into a list (one per GPU)
            gpu_temps = temp.strip().split('\n')
            
            # Get the temperature for the current GPU (based on local_rank)
            local_rank = accelerator.local_process_index
            if local_rank < len(gpu_temps):
                gpu_temp = int(gpu_temps[local_rank])
                print(f"GPU {local_rank} temperature: {gpu_temp}°C")  # Print temperature
                if gpu_temp > max_temp:
                    print(f"GPU {local_rank} temperature is too high: {gpu_temp}°C. Pausing training.")
                    return False
            else:
                print(f"No temperature data for GPU {local_rank}. Skipping temperature check.")
        return True
    except Exception as e:
        print(f"Temperature check failed: {e}. Skipping check.")
        return True
    
# Loading a YAML file
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
    
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
    checkpoint_dir="checkpoints",
    resume_training=False
):
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

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
       
        # Add GPU memory info to logging
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"Initial GPU Memory: {allocated:.2f}GB/{reserved:.2f}GB")
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        vit.parameters(),
        lr=max_lr/25.,  # base learning rate
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Initialize training metrics
    start_epoch = 0
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    no_improve = 0

    # Resume from checkpoint if requested
    if resume_training:
        latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            logger.info(f"Resuming from checkpoint: {latest_checkpoint}")
            checkpoint = load_checkpoint(latest_checkpoint, vit, optimizer, None)  # Don't load scheduler state
            if checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                train_losses = checkpoint['train_losses']
                val_losses = checkpoint['val_losses']
                best_val_loss = checkpoint['best_val_loss']
                no_improve = checkpoint['no_improve']
                logger.info(f"Resuming from epoch {start_epoch}")
            else:
                logger.warning("No checkpoint found, starting from scratch")
        else:
            logger.warning("No checkpoint found, starting from scratch")
            
    # Calculate remaining steps and initialize scheduler
    remaining_epochs = num_epochs - start_epoch
    total_steps = len(train_loader) * remaining_epochs

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
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        vit.train()
        running_train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            with accelerator.accumulate(vit):
                optimizer.zero_grad()
                outputs = vit(images)
                loss = F.cross_entropy(outputs, targets, label_smoothing=label_smoothing)
                
                # Backward pass with accelerator
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    clip_grad_norm_(vit.parameters(), max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                
                running_train_loss += loss.item()
                num_batches += 1
        
        # Calculate average training loss for the epoch
        train_loss = running_train_loss / num_batches
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
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
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
                
        if accelerator.is_main_process and torch.cuda.is_available():
            print_gpu_memory()
            torch.cuda.empty_cache()
        
        # Save checkpoint
        save_checkpoint(
            accelerator,
            vit,
            optimizer,
            scheduler,
            epoch,
            train_losses,
            val_losses,
            best_val_loss,
            no_improve,
            checkpoint_dir
        )
        
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

if __name__ == "__main__":
    
    config = load_config('./config/vit_fmnist.yaml')
    
    ## Dataset params
    dir = config['dataset_params']['dir']
    batch_size = config['dataset_params']['batch_size']
    n_workers = config['dataset_params']['n_workers']
    ### 
    print(f"Dataset directory: {dir}")
    print(f"Batch size: {batch_size}")
    print(f"Number of workers: {n_workers}")
    
    ## Model params
    channels = config['model_params']['channels']
    image_height = config['model_params']['image_height']
    image_width = config['model_params']['image_width']
    patch_height = config['model_params']['patch_height']
    patch_width = config['model_params']['patch_width']
    emb_dim = config['model_params']['emb_dim']
    n_heads = config['model_params']['n_heads']
    n_layers = config['model_params']['n_layers']
    hidden_dim = config['model_params']['hidden_dim']
    dropout = config['model_params']['dropout']
    n_classes = config['model_params']['n_classes']
    ###
    print(f"Number of channels (C): {channels}")
    print(f"Height (H): {image_height}")
    print(f"Width (W): {image_width}")
    print(f"Patch size (patch_size): {patch_height}")
    # print(f"Number of patches (num_patches): {num_patches}")
    print(f"Embedding dimension (emb_dim): {emb_dim}")
    print(f"Number of heads (num_heads): {n_heads}")
    print(f"Number of transformer layers (num_layers): {n_layers}")
    print(f"Hidden dimensions: {hidden_dim}")
    print(f"Dropout: {dropout}")   
    print(f"Number of classes: {n_classes}")
    # print(f"Head dimension: {emb_dim // num_heads}")
    
    ## Training params
    n_epochs = config['training_params']['n_epochs']
    max_lr = config['training_params']['max_lr']
    max_grad_norm = config['training_params']['max_grad_norm']
    weight_decay = config['training_params']['weight_decay']
    patience = config['training_params']['patience']
    label_smoothing = config['training_params']['label_smoothing']
    ### 
    print(f"Number of epochs: {n_epochs}")
    print(f"Max lr: {max_lr}")
    print(f"Max grad norm: {max_grad_norm}")
    print(f"Label smoothing: {label_smoothing}")
    
    ## Eval params
    val_freq = config['eval_params']['val_freq']
    ### 
    print(f"Evaluation frequency: {val_freq}")

    # Create data transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(image_height, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3)  # Add random erasing
    ])

    val_transform = transforms.Compose([
        transforms.Resize(image_height),
        transforms.ToTensor(),
    ])

    # # Create datasets
    ds_train = torchvision.datasets.FashionMNIST(
        root=dir, 
        train=True,
        transform=train_transform,
        download=True
    )

    ds_val = torchvision.datasets.FashionMNIST(
        root=dir, 
        train=False,
        transform=val_transform,
        download=True
    )

    ## Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        ds_train, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=n_workers, 
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        ds_val, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=n_workers
    )

    ## Initialize model
    vit = VIT(
        num_layers=n_layers,
        num_heads=n_heads,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        patch_size=patch_height,
        C=channels,
        img_shape=(image_height, image_width),
        n_classes=n_classes,
        dropout=dropout,
    )
    
    ## Compile model
    vit = torch.compile(vit)

    ## Calculate and print number of parameters
    total_params = 0
    for name, param in vit.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            # print(f"{name}: {num_params:,} parameters")
            total_params += num_params
    print(f"Total trainable parameters: {total_params:,}")

    # # Train the model
    train_losses, val_losses = train_vit(
        vit=vit,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=n_epochs,
        max_lr=max_lr,
        max_grad_norm=max_grad_norm,
        weight_decay=weight_decay,
        val_freq=1,
        patience=patience,
        checkpoint_dir="checkpoints",
        resume_training=True  # Set to True to resume from latest checkpoint
    )