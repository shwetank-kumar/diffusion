## How VIT works in simple english
* Take an image say its (16, 32, 32, 3) i.e. height and width are 32 pixels each with 3 channels (C). We will also assume a batch size of 16 so this is closer to what it will look like in code

## Turning image batch into patches
* This is to convert images into a format that will be accepted by a Transformer architecture - nominally a sequence of tokens which then we layer on top positional information. To do this we start by converting images into patches. 
* Lets say patch_size for height and width is same and is 8 pixels
    * We have (4, 4, 3) patches each with (8, 8) pixels
* As we turn these image patches into sequence of tokens we need to be thoughtful about how to preserve spatial information (think how to flatten this structure within transformer context length)
* To convert this to series of tokens you take each path and flatten it to get:
    * Patch 1: R1, G1, B1, R2, G2, B2 ... R64, G64, B64 -> 192 flattened pixels in a linear vector
    * Patch 2: R1, G1, B1, R2, G2, B2 ... R64, G64, B64 -> 192 flattened pixels in a linear vector
    * Patch 3: R1, G1, B1, R2, G2, B2 ... R64, G64, B64 -> 192 flattened pixels in a linear vector
    * Patch 4: R1, G1, B1, R2, G2, B2 ... R64, G64, B64 -> 192 flattened pixels in a linear vector
    * So here we have batch (16), num_patches (16), patch_size x patch_size x C (192 = 8 x 8 x 3)  
    * Once these operations are done we can start projecting this tensor into embedding dimension of the transformer so we get (16, 16, emb_dim=764) by just passing the previous tensor through a linear layer of spec (192, 764).
    * For each image we also have a learnable CLS token which we prepend at the very top to get a tensor:
    batch (16), 1 + num_patches (17), patch_size (192 = 8 x 8 x 3)
    * We now pre-pend a class token at the top to get: (16, 17, 764)
    * Our positional encoding layer will add to this so it will be a parameter layer of (1, 17, 764)
    * So at the end of this step we have (16, 17, 764) tensor this can go into the transformer

## Feeding patached images as tokens into the transformer layer
* Transformer essentially works on the embedding dimension which here is 764. You basically put the token of dimension (batch_size, num_patches + 1, emb_dim) into linear layers of wk, wq, wv each of which are (emb_dim, emb_dim)  --> (batch_size, num_patches + 1, emb_dim)
* Now you reshape these into heads so you take q, k, v which are of shape (batch_size, num_patches + 1, emb_dim) and reshape it into (batch_size, num_patches + 1, num_heads, head_size)
* Then you calculate attention weights as: q @ torch.transpose(k, -1, -2) // head_dim ** 0.5 (scaled dot product attention)  --> this will have shape (batch_size, num_patches + 1, num_heads, num_heads)
* You convert attention weights into attention scores by taking softmax torch.softmax(attention_scores, dim=-1) --> this will have shape (batch_size, num_patches + 1, num_heads, num_heads)
* output = attention_scores @ v --> this will have shape (batch_size, num_patches + 1, num_heads, head_size)
* We then concatenate all the heads together to get back emb_dim so output is --> (batch_size, num_patches + 1, emb_dim)
    