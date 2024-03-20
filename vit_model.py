import torch
import torch.nn as nn
import numpy as np
from transformer_encoder import TransformerEncoder

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, channels, encoder_blocks, n_heads,token_dim, mlp_dim, pos_encoding_learnable):
        super(ViT, self).__init__()
        # Currently only supports square images
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
       
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        assert num_patches * patch_dim == channels * image_size ** 2, 'Patches must tile the image exactly.'
        
        self.patch_size = patch_size

        # Linear projection of the flattened patches (learnable)
        self.linear_projection = nn.Linear(patch_dim, token_dim)

        # Token for the class (learnable)
        self.class_token = nn.Parameter(torch.randn(1, token_dim))

        # Positional embedding can be learned or fixed (+1 for the class token)
        self.pos_embedding = self._get_pos_embedding(num_patches + 1, token_dim, pos_encoding_learnable)
        
        # Transformer Encoder (we can have multiple encoders)
        self.transformers = nn.ModuleList([TransformerEncoder(token_dim, n_heads, mlp_dim) for _ in range(encoder_blocks)])
        
        self.mlp_head = nn.Sequential(
            nn.Linear(token_dim, num_classes),
        )


    def forward(self, image):

        patches = self._make_patches(image)
        tokens = self.linear_projection(patches)

        batch_size = tokens.size(0)
        # append class token to the beginning of the sequence for each image in the batch
        tokens = torch.stack([torch.vstack([self.class_token, tokens[i]]) for i in range(batch_size)])

        # add positional embedding to the patches (only repeats batch_size times, other dimensions are broadcasted automatically)
        pos_embedding = self.pos_embedding.repeat(batch_size, 1, 1)

        tokens += pos_embedding

        # Pass through Transformer Encoder (we can have multiple encoders)
        for transformer in self.transformers:
            tokens = transformer(tokens)

        # Taking only the classification token
        input = tokens[:, 0]

        # Running through the classification network
        out = self.mlp_head(input)

        return out
    
    def _make_patches(self, image):
        # input is a tensor of shape (B, C, H, W)
        # B = batch size, C = n_channels, H = height, W = width
        p = self.patch_size

        # makes patches of shape along height (B, C, H, W) -> (B, C, H/p, W, p)
        x = image.unfold(2, p, p)

        # makes patches of shape along width (B, C, H/p, W, p) -> (B, C, H/p, W/p, p, p)
        x = x.unfold(3, p, p)
        
        # makes patches of shape (B, C, H/p, W/p, p, p) -> (B, C, H/p * W/p, p * p)
        x = x.contiguous().view(x.size(0), x.size(1), -1, p * p)
        
        # flattens channels (B, C, H/p * W/p, p * p) -> (B, H/p * W/p, C * p * p)
        x = x.contiguous().view(x.size(0), x.size(2), -1)
        
        return x
    

    def _get_pos_embedding(self, n_tokens, dim, learnable):
        if learnable:
            # uses learnable positional embeddings
            return nn.Parameter(torch.randn(n_tokens, dim))
        else:
            # uses fixed positional embeddings
            return nn.Parameter(self._fixed_positional_embedding(n_tokens, dim).clone().detach(), requires_grad=False)
    
    # positional embedding proposed in the 'Attention is All You Need' paper
    def _fixed_positional_embedding(self, n_tokens, dim):
        result = torch.zeros(n_tokens, dim)
        for i in range(n_tokens):
            for j in range(dim):
                if j % 2 == 0:
                    result[i , j] = np.sin(i / 10000 ** (j / dim))
                else:
                    result[i, j] = np.cos(i / 10000 ** ((j - 1) / dim))

        return result

    