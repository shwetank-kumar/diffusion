import torch
from torch import nn
from einops import rearrange, repeat
import torch.nn.functional as F

class Patchify(nn.Module):
    def __init__(self, patch_size: int, C: int, img_shape: tuple, emb_dim: int, dropout: float):
        """
        Args:
            patch_size (int): Size of each patch (height and width).
            C (int): Number of input channels.
            img_shape (tuple): Shape of the input image (height, width).
            emb_dim (int): Dimension of the patch embeddings.
            dropout (float): Dropout rate.
        """
        super(Patchify, self).__init__()
        self.patch_size = patch_size
        self.C = C
        self.h, self.w = img_shape
        self.emb_dim = emb_dim
        
        # Calculate the number of patches
        self.num_patches = (self.h // patch_size) * (self.w // patch_size)
        
        # Learnable class token (shape: (1, emb_dim))
        self.class_token = nn.Parameter(torch.randn(1, emb_dim))
        
        # Learnable positional embeddings (shape: (1, num_patches + 1, emb_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches + 1, emb_dim))
        
        
        # Linear projection layer with sequential norm and projection
        self.proj = nn.Linear(patch_size * patch_size * C, emb_dim)
        
        self.patch_emb_dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patchify the image
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        
        # Project patches
        x = self.proj(x)
        
        # Get batch size dynamically
        batch_size = x.shape[0]
        
        # Expand class token
        class_token = repeat(self.class_token, '1 d -> b 1 d', b=batch_size)
        
        # Prepend class token
        x = torch.cat([class_token, x], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_emb
        
        # Apply dropout
        x = self.patch_emb_dropout(x)
        
        return x

class MHSA(nn.Module):
    def __init__(self, num_heads, emb_dim, dropout):  # Add dropout parameter
        super().__init__()
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // num_heads
        
        # Combine QKV into single matrix for efficiency
        self.qkv = nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, num_patches_plus_1, emb_dim = x.shape
        
        # Single matrix multiplication for Q, K, V
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b n (three h d) -> three b h n d', 
                       three=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Use PyTorch's native scaled dot-product attention
        output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=False
        )
        
        # Reshape output
        output = rearrange(output, 'b h n d -> b n (h d)')
        output = self.proj(output)
        output = self.proj_dropout(output)
        
        return output


class VITLayer(nn.Module):
    def __init__(self, num_heads, emb_dim, hidden_dim, dropout):
        super(VITLayer, self).__init__()
        self.attention_norm = nn.LayerNorm(emb_dim)
        self.ff_norm = nn.LayerNorm(emb_dim)
        
        self.attention_block = MHSA(num_heads, emb_dim, dropout)
        self.ff_block = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout)
        )
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        # x = x + self.drop_path(self.attention_block(self.attention_norm(x)))
        # x = x + self.drop_path(self.ff_block(self.ff_norm(x)))
        x = x + self.attention_block(self.attention_norm(x))
        x = x + self.ff_block(self.ff_norm(x))
        return x

class VIT(nn.Module):
    def __init__(self, num_layers, num_heads, emb_dim, hidden_dim, patch_size,
                 C, img_shape, n_classes, dropout=0.1):# drop_path_rate=0.1):
        super(VIT, self).__init__()
        self.patch_embedding = Patchify(patch_size=patch_size, C=C, 
                                      img_shape=img_shape, emb_dim=emb_dim, 
                                      dropout=dropout)
        
        # Initialize drop path rates (linearly increasing)
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.xfls = nn.ModuleList([
            VITLayer(
                num_heads=num_heads,
                emb_dim=emb_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                # drop_path=dpr[i]
            ) for i in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(emb_dim)
        # self.lin = nn.Linear(emb_dim, n_classes)
        self.pre_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh(),
        )
        self.head = nn.Linear(emb_dim, n_classes)
        
    def forward(self, x, return_features=False):
        x = self.patch_embedding(x)
        for layer in self.xfls:
            x = layer(x)
        x = self.norm(x)
        x = x[:, 0]  # Get CLS token
        features = self.pre_head(x)
        logits = self.head(features)
        
        if return_features:
            return features, logits
        return logits