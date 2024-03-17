# implementation of the encoder part of the transformer model
import torch
import torch.nn as nn
import numpy as np

# Multi-head self-attention block
class MHA(nn.Module):
    def __init__(self, token_dim, n_heads):
        super(MHA, self).__init__()

        assert token_dim % n_heads == 0, f'Token dimension ({token_dim}) must be divisible by the number of heads({n_heads}).'
        head_dim = token_dim // n_heads

        self.head_dim = head_dim
        self.n_heads = n_heads

        self.query = nn.ModuleList([nn.Linear(head_dim, head_dim) for _ in range(n_heads)])
        self.key = nn.ModuleList([nn.Linear(head_dim, head_dim) for _ in range(n_heads)])
        self.value = nn.ModuleList([nn.Linear(head_dim, head_dim) for _ in range(n_heads)])

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, token_seq):
        # B = batch size, T = sequence length, TD = token dimension
        # H = number of heads, HD = head dimension
        # token_seq shape: (B, T, TD)
        batch_size, seq_len, token_dim = token_seq.shape
        
        # reshape the token sequence to split the token dimension into n_heads
        # (B, T, TD) -> (B, T, H, HD)
        token_seq = token_seq.reshape(batch_size, seq_len, self.n_heads, self.head_dim)

         # calculate self-attention for each head
        for i in range(self.n_heads):
            # query, key, value have the shape -> (B, T, HD)
            query = self.query[i](token_seq[:, :, i])
            key = self.key[i](token_seq[:, :, i])
            value = self.value[i](token_seq[:, :, i])

            # calculate the attention scores
            # attention shape: (B, T, T)
            attention = self.softmax(torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.head_dim))

            # apply the attention scores to the value
            # out shape: (B, T, HD)
            out = torch.matmul(attention, value)

            # concatenate the outputs of each head
            if i == 0:
                concat_out = out
            else:
                concat_out = torch.cat((concat_out, out), dim=-1)

        # concatenated output is the same shape as input -> (B, T, TD)
        return concat_out
    

class TransformerEncoder(nn.Module):
    def __init__(self, token_dim, n_heads, mlp_dim):
        super(TransformerEncoder, self).__init__()
        self.token_dim = token_dim
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(token_dim)
        self.mha = MHA(token_dim, n_heads)
        self.norm2 = nn.LayerNorm(token_dim)

        self.mlp = nn.Sequential(
            nn.Linear(token_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, token_dim)
        )

    def forward(self, input):
        # B = batch size, T = sequence length, TD = token dimension
        # input is of shape -> (B, T, TD)       
        l_norm1 = self.norm1(input) 
        out = input + self.mha(l_norm1)
        l_norm2 = self.norm2(out)
        out_mlp = out + self.mlp(l_norm2)

        # output retains the input shate (B, T, TD)
        return out_mlp
         