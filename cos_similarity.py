import torch
from vit_model import ViT
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from utils import read_config


def plot_pos_embedding(pos_emb, num_patches):
    
    # Calculate the cosine similarity between each pair of patch positions
    cosine_sim = torch.nn.functional.cosine_similarity(pos_emb.unsqueeze(1), pos_emb.unsqueeze(0), dim=-1)

    # Plotting the grid of cosine similarity matrices
    fig, axes = plt.subplots(num_patches, num_patches, figsize=(10, 10))

    for i in range(num_patches):
        for j in range(num_patches):
            ax = axes[i, j]
            patch_index = i * num_patches + j
            
            # Plot the heatmap
            sns.heatmap(cosine_sim[patch_index].reshape(num_patches, num_patches).detach().numpy(), 
                        annot=False, cmap='viridis', cbar=False, square=True, ax=ax)

            # Remove ticks and labels from all plots
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Only add labels to the leftmost column and bottom row
            if j == 0:
                ax.set_ylabel(f'{i + 1}', fontsize=12, rotation=0, labelpad=15)
            if i == num_patches - 1:
                ax.set_xlabel(f'{j + 1}', fontsize=12)

    # Add a single colorbar to the right of the last column
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Position: [left, bottom, width, height]
    # Create heatmap for colorbar
    sns.heatmap(cosine_sim[0].reshape(num_patches, num_patches).detach().numpy(), cmap='viridis', cbar=True, ax=axes[0, 0], cbar_ax=cbar_ax, annot=False, square=True)
    
    
    # Add title to the right of the colorbar
    fig.text(0.98, 0.5, 'Kosinusna sličnost', rotation=270, va='center', ha='center', fontsize=14)

    # Specifically clear ticks and labels from the first patch
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    axes[0, 0].tick_params(axis='both', which='both', length=0)
    axes[0, 0].set_xlabel('')
    axes[0, 0].set_ylabel('1', fontsize=12, rotation=0, labelpad=15)

    # Set overall title and labels
    fig.suptitle('Kosinusna sličnost pozicionih vektora različitih patch-eva', fontsize=16)
    fig.text(0.5, 0.05, 'x-koordinata patch-a', ha='center', va='center', fontsize=14)
    fig.text(0.05, 0.5, 'y-koordinata patch-a', ha='center', va='center', rotation='vertical', fontsize=14)
    plt.subplots_adjust(top=0.93, right=0.9, wspace=0.1, hspace=0.1)

    plt.show()



class SimilarityPlot():

    def __init__(self, config_file='hyper_params.json', pretrained_model=False):
        # read hyperparameters from config file
        config = read_config(config_file)
        self.set_config(config)
        self.set_model(pretrained_model)

        # Generate the positional embeddings
        self.pos_emb = self.model.pos_embedding


# # Generate the positional embeddings
# pos_emb = model.pos_embedding

    def set_config(self, config):
        self.patch_size = config['patch_size']
        self.pos_encoding_learnable=config['pos_encoding_learnable']
        self.token_dim=config['token_dim']
        self.n_heads=config['n_heads']
        self.encoder_blocks=config['encoder_blocks']
        self.mlp_dim=config['mlp_dim']
        self.batch_size =config['batch_size']
        self.n_epochs =config['n_epochs']
        self.lr=config['lr']
        self.best_path = config['best_path']

        # Constants
        self.image_size = 28
        self.channels = 1
        self.num_classes = 10

        self.num_patches = (self.image_size // self.patch_size) # Example: 4x4 grid of patches
    
    
    def set_model(self, pretrained_model):
        self.model = ViT(image_size=self.image_size, patch_size=self.patch_size,
        num_classes=self.num_classes, channels=self.channels,
        pos_encoding_learnable=self.pos_encoding_learnable,
        token_dim=self.token_dim, mlp_dim=self.mlp_dim,
        encoder_blocks=self.encoder_blocks, n_heads=self.n_heads)

        if pretrained_model:
            self.model.load_state_dict(torch.load(self.best_path))    
        


if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Plotting cos similarity.')
    parser.add_argument('--pretrained', action='store_true', default=False, 
                    help='Flag to use a pretrained model (default is False)')

    # Parse arguments
    args = parser.parse_args()

    sim = SimilarityPlot(pretrained_model=args.pretrained)
    plot_pos_embedding(sim.pos_emb, sim.num_patches)