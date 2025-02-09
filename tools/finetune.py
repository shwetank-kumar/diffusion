import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import timm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import logging
from pathlib import Path
from tqdm import tqdm

logger = get_logger(__name__)

def create_model(num_classes=100):
    """Load pre-trained ViT-Base and modify head for CIFAR-100"""
    # Load pre-trained ViT-Base/16 from timm
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    
    # Replace classification head
    model.head = nn.Linear(model.head.in_features, num_classes)
    
    # Initialize the new head with zeros as per paper
    nn.init.zeros_(model.head.weight)
    nn.init.zeros_(model.head.bias)
    
    return model

def create_transforms(img_size=224):
    """Create transforms for fine-tuning"""
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_epoch(model, loader, criterion, optimizer, scheduler, accelerator, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(loader, desc=f'Epoch {epoch+1}')
    
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        with accelerator.accumulate(model):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                # Gradient clipping as per paper
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar.set_postfix({
                'loss': total_loss/(batch_idx+1),
                'acc': 100.*correct/total,
                'lr': optimizer.param_groups[0]['lr']
            })
    
    return total_loss/len(loader), 100.*correct/total

def validate(model, loader, criterion, accelerator):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss/len(loader), 100.*correct/total

def main():
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=1,
        log_with="tensorboard",
        project_dir="finetune_logs"
    )
    
    if accelerator.is_main_process:
        Path("checkpoints").mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    set_seed(42)
    
    # Create model
    model = create_model(num_classes=100)  # CIFAR-100 has 100 classes
    
    # Create transforms and load CIFAR-100
    train_transform, val_transform = create_transforms()
    
    train_dataset = datasets.CIFAR100(
        root='./datasets',
        train=True,
        download=True,
        transform=train_transform
    )
    
    val_dataset = datasets.CIFAR100(
        root='./datasets',
        train=False,
        download=True,
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,  # Adjust based on your GPU memory
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    
    # Lower learning rate for fine-tuning
    optimizer = AdamW(
        model.parameters(),
        lr=1e-4,  # Lower learning rate for fine-tuning
        weight_decay=0.1  # Slightly lower weight decay for fine-tuning
    )
    
    # Calculate steps for scheduler
    num_epochs = 30  # Fewer epochs for fine-tuning
    total_steps = len(train_loader) * num_epochs
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-4,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    # Prepare for distributed training
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    # Training loop
    best_acc = 0
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, 
            accelerator, epoch
        )
        
        val_loss, val_acc = validate(model, val_loader, criterion, accelerator)
        
        if accelerator.is_main_process:
            logger.info(
                f'Epoch: {epoch+1}/{num_epochs} | '
                f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}% | '
                f'Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f}%'
            )
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                accelerator.save(
                    model.state_dict(),
                    f"checkpoints/best_model.pth"
                )
                logger.info(f'New best accuracy: {best_acc:.2f}%')

if __name__ == "__main__":
    main()