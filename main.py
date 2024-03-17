import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from vit_model import ViT


#########################
# to be set in a config file
batch_size = 32

#########################

# Define transformations to apply to the data
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image data
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create data loaders
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# Define the classes of the dataset
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {torch.cuda.get_device_name(device)}" if device.type == "cuda" else "Device: CPU")

# Define the model
image_size = 32
channels = 3
patch_size = 16
num_classes = 10

model = ViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, channels=channels, pos_encoding_learnable=False)
image_batch = next(iter(trainloader))[0]
output = model(image_batch)
# print(output.shape)
