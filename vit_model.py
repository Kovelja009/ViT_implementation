import torch
import torch.nn as nn
import numpy as np

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, channels, token_dim=512, pos_encoding_learnable=False):
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
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.patch_to_embedding = nn.Linear(patch_dim, dim)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim), num_layers=depth)
        # self.to_cls_token = nn.Identity()
        # self.mlp_head = nn.Sequential(
        #     nn.Linear(dim, mlp_dim),
        #     nn.GELU(),
        #     nn.Linear(mlp_dim, num_classes)
        # )


    def forward(self, image):

        patches = self._make_patches(image)
        tokens = self.linear_projection(patches)

        batch_size = tokens.size(0)
        # append class token to the beginning of the sequence for each image in the batch
        tokens = torch.stack([torch.vstack([self.class_token, tokens[i]]) for i in range(batch_size)])

        # add positional embedding to the patches (only repeats batch_size times, other dimensions are broadcasted automatically)
        pos_embedding = self.pos_embedding.repeat(batch_size, 1, 1)

        tokens += pos_embedding


        return tokens
        # x = self.patch_to_embedding(x)
        # x = x.view(x.size(0), -1, x.size(-1))
        # cls_tokens = self.cls_token.expand(image.size(0), -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding
        # x = self.transformer(x)
        # x = self.to_cls_token(x[:, 0])
        # return self.mlp_head(x)
    
    def _make_patches(self, image):
        # input is a tensor of shape (B, C, H, W)
        # B = batch size, C = n_channels, H = height, W = width
        p = self.patch_size

        # makes patches of shape along height (B, C, H, W) -> (B, C, H/p, W, p)
        x = image.unfold(2, p, p)
        print(f'{x.shape}')

        # makes patches of shape along width (B, C, H/p, W, p) -> (B, C, H/p, W/p, p, p)
        x = x.unfold(3, p, p)
        print(f'{x.shape}')
        
        # makes patches of shape (B, C, H/p, W/p, p, p) -> (B, C, H/p * W/p, p * p)
        x = x.contiguous().view(x.size(0), x.size(1), -1, p * p)
        print(f'{x.shape}')
        
        # flattens channels (B, C, H/p * W/p, p * p) -> (B, H/p * W/p, C * p * p)
        x = x.contiguous().view(x.size(0), x.size(2), -1)
        print(f'{x.shape}')
        
        return x
    

    def _get_pos_embedding(self, n_tokens, dim, learnable):
        if learnable:
            # uses learnable positional embeddings
            return nn.Parameter(torch.randn(n_tokens, dim))
        else:
            # uses fixed positional embeddings proposed
            return nn.Parameter(torch.tensor(self._fixed_positional_embedding(n_tokens, dim)), requires_grad=False)
    
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

    