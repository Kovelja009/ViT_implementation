import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from vit_model import ViT
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from utils import save_params_to_json, read_config
import time

torch.manual_seed(0)

def training():

    #########################
    # read hyperparameters from config file
    config = read_config('hyper_params.json')
    patch_size = config['patch_size']
    pos_encoding_learnable=config['pos_encoding_learnable']
    token_dim=config['token_dim']
    n_heads=config['n_heads']
    encoder_blocks=config['encoder_blocks']
    mlp_dim=config['mlp_dim']
    batch_size =config['batch_size']
    n_epochs =config['n_epochs']
    lr=config['lr']
    best_path = config['best_path']

    # Constants
    image_size = 28
    channels = 1
    num_classes = 10
    log_filename = 'logs/fixed_embeddings_without_class_token.json'
    #########################

    # Define transformations to apply to the data
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL Image to tensor
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image data
    ])

    # Load MNIST dataset
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

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
                pos_encoding_learnable=pos_encoding_learnable,
                token_dim=token_dim, mlp_dim=mlp_dim,
                encoder_blocks=encoder_blocks, n_heads=n_heads).to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    loss = CrossEntropyLoss()

    epoch_times = []
    training_losses = []
    testing_losses = []
    testing_accuracies = []

    for epoch in tqdm(range(n_epochs)):
        model.train()
        train_loss = 0

        # Start epoch timer
        start_time = time.time()

        for i, data in tqdm(enumerate(trainloader), desc=f'Epoch {epoch + 1} training', leave=False):
            optimizer.zero_grad()

            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = model(inputs)
            loss_value = loss(outputs, labels)
            loss_value.backward()
            optimizer.step()

            train_loss += loss_value.detach().cpu().item()

        
        end_time = time.time()
        epoch_times.append(end_time - start_time)

        t_loss = train_loss / len(trainloader)
        training_losses.append(t_loss)
        print(f'Epoch {epoch + 1} training loss: {t_loss:.4f}')


        model.eval()
        test_loss = 0
        correct, total = 0, 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(testloader), desc=f'Epoch {epoch + 1} testing', leave=False):
                inputs, labels = data[0].to(device), data[1].to(device)

                outputs = model(inputs)
                loss_value = loss(outputs, labels)

                # calculate loss
                test_loss += loss_value.detach().cpu().item()
                
                # calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        t_loss = test_loss / len(testloader)
        testing_losses.append(t_loss)
        testing_accuracies.append(test_accuracy)
        print(f'Epoch {epoch + 1} testing accuracy: {test_accuracy:.2f}%')
        print(f'Epoch {epoch + 1} testing loss: {t_loss:.4f}')


    print('Finished Training')

    # Save the model
    print(f'Saving model to {best_path}')
    torch.save(model.state_dict(), best_path)

    print(f'Saving session to {log_filename}')
    save_params_to_json(log_filename,
                        config={
                            "patch_size": patch_size,
                            "pos_encoding_learnable": pos_encoding_learnable,
                            "token_dim": token_dim,
                            "n_heads": n_heads,
                            "encoder_blocks": encoder_blocks,
                            "mlp_dim": mlp_dim,
                            "batch_size": batch_size,
                            "lr": lr,
                        },
                        metrics={
                            "training_losses": training_losses,
                            "testing_losses": testing_losses,
                            "times_per_epoch": epoch_times,
                            "testing_accuracies": testing_accuracies
                        })




if __name__ == "__main__":
    training()
