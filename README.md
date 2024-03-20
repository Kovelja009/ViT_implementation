# Vision Transformer in PyTorch

<div style="text-align:center;">
  <img src="https://github.com/Kovelja009/ViT_impementation/assets/81018289/38ff49cf-a7b3-4385-91d4-0d48006db69d" alt="Description of the image" width="500">
</div>

This repository contains an implementation of the **Vision Transformer** (**ViT**) model using PyTorch. ViT applies the transformer architecture, originally designed for natural language processing tasks, to computer vision tasks.
## Overview:

1. **Patching**:
  - The input image is divided into non-overlapping patches of fixed size.
  - Each patch is typically square-shaped and contains a fixed number of pixels.
  - For example, common patch sizes are 16x16 or 32x32 pixels.
  - The image is divided into patches in a grid-like fashion, preserving spatial information.

2. **Patch Embedding**:

  - After dividing the image into patches, each patch is flattened into a one-dimensional vector.
  - These patch vectors are then passed through an embedding layer to project them into a lower-dimensional space.
  - The embedding layer transforms each patch vector into a continuous representation suitable for processing by the transformer model.
  - This embedding process allows the model to learn meaningful representations of image patches.

3. **Tokenization**:

  - In addition to the patch embeddings, a learnable "class" token is added to represent the entire image.
  - This class token is concatenated with the embeddings of individual patches.
  - Together, the patch embeddings and the class token form the input tokens for the transformer model.
  - The sequence of tokens is then fed into the transformer encoder for further processing.

4. **Transformer Encoding**:

  - The tokenized input sequence, consisting of patch embeddings and the class token, is passed through multiple layers of transformer encoders.
  - Each transformer encoder layer applies self-attention mechanisms and feedforward neural networks to capture global and local dependencies within the image.
  - The output of the transformer encoder represents the high-level semantic features of the image, which can be used for downstream tasks such as image classification or object detection.

5. **Classification Network**:
   
   - After processing the input image through the transformer encoder, the output of the encoder consists of a sequence of token embeddings, where each token represents a patch of the image.
   - To perform image classification, typically only the embedding corresponding to the "class" token (i.e., the first token) is used.
   - This token embedding is passed through a classification network, which may consist of fully connected layers, convolutional layers, or a combination of both.
   - The classification network outputs logits for each class, which are then passed through a softmax layer to obtain class probabilities.

In summary, **ViT** model transforms the input image into a sequence of tokens that are appended to the **class token** that are then passed through the **transformer encoder**, with the class token at the output of encoder representing the input for the **classification network**.

## Features
- Implementation of the Vision Transformer model in PyTorch.
- Training and evaluation scripts for image classification tasks.

## Usage
1. Install dependencies:
```
pip install -r requirements.txt
```
2. Run the training script
```
python main.py
```


## References
- Original paper: [Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.](https://arxiv.org/abs/2010.11929)

## Credits
This implementation is based on the paper and various open-source resources. Special thanks to the authors and contributors for their work.
