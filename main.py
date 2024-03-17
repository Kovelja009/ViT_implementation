import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from vit_model import ViT
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

torch.manual_seed(0)

#########################
# to be set in a config file
image_size = 32
channels = 3
patch_size = 4
num_classes = 10
batch_size = 512
n_epochs = 10
lr= 0.01
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


# Create the model and move it to the device (GPU if available)
model = ViT(image_size=image_size, patch_size=patch_size,
            num_classes=num_classes, channels=channels,
              pos_encoding_learnable=False,
              token_dim=64, mlp_dim=32,
              encoder_blocks=1, n_heads=8).to(device)

optimizer = Adam(model.parameters(), lr=lr)
loss = CrossEntropyLoss()

for epoch in tqdm(range(n_epochs)):
    model.train()
    train_loss = 0
    for i, data in tqdm(enumerate(trainloader), desc=f'Epoch {epoch + 1} training', leave=False):
        optimizer.zero_grad()

        inputs, labels = data[0].to(device), data[1].to(device)

        outputs = model(inputs)
        loss_value = loss(outputs, labels)
        loss_value.backward()
        optimizer.step()

        train_loss += loss_value.detach().cpu().item()

    print(f'Epoch {epoch + 1} training loss: {train_loss / len(trainloader):.4f}')

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), desc=f'Epoch {epoch + 1} testing', leave=False):
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = model(inputs)
            loss_value = loss(outputs, labels)

            test_loss += loss_value.detach().cpu().item()

    print(f'Epoch {epoch + 1} testing loss: {test_loss / len(testloader):.4f}')

print('Finished Training')


