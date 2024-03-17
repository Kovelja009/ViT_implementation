# implementation of the encoder part of the transformer model
import torch
import torch.nn as nn
import numpy as np

# Multi-head self-attention 
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
        print("############ MHA forward ############")
        print(f'batch_size: {batch_size}, seq_len: {seq_len}, token_dim: {token_dim}')
        
        # reshape the token sequence to split the token dimension into n_heads
        # (B, T, TD) -> (B, T, H, HD)
        token_seq = token_seq.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        print(f'token_seq reshaped: {token_seq.shape}')


         # calculate self-attention for each head
        for i in range(self.n_heads):
            # query, key, value have the shape -> (B, T, HD)
            query = self.query[i](token_seq[:, :, i])
            key = self.key[i](token_seq[:, :, i])
            value = self.value[i](token_seq[:, :, i])
            print(f'query: {query.shape}, key: {key.shape}, value: {value.shape}')

            # calculate the attention scores
            # attention shape: (B, T, T)
            attention = self.softmax(torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.head_dim))
            print(f'attention: {attention.shape}')

            # apply the attention scores to the value
            out = torch.matmul(attention, value)
            # out shape: (B, T, HD)
            print(f'out: {out.shape}')

            # concatenate the outputs of each head
            if i == 0:
                concat_out = out
            else:
                concat_out = torch.cat((concat_out, out), dim=-1)

        # concatenated output is the same shape as input -> (B, T, TD)
        print(f'concat_out: {concat_out.shape}')

        return concat_out