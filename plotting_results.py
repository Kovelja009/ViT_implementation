import json
import matplotlib.pyplot as plt

def plot_losses(filename1, filename2, title):
    model1 = filename1.split('/')[-1].split('.')[0]
    model2 = filename2.split('/')[-1].split('.')[0]
    with open(filename1, 'r') as f1, open(filename2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    epochs = list(range(1, len(data1['metrics']['training_losses']) + 1))
    training_losses1 = data1['metrics']['training_losses']
    testing_losses1 = data1['metrics']['testing_losses']
    training_losses2 = data2['metrics']['training_losses']
    testing_losses2 = data2['metrics']['testing_losses']
    
    plt.plot(epochs, training_losses1, label=f'Training Loss for {model1}')
    plt.plot(epochs, testing_losses1, label=f'Test Loss for {model1}')
    plt.plot(epochs, training_losses2, label=f'Training Loss for {model2}')
    plt.plot(epochs, testing_losses2, label=f'Test Loss for {model2}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Losses')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracies(filename1, filename2, title):
    model1 = filename1.split('/')[-1].split('.')[0]
    model2 = filename2.split('/')[-1].split('.')[0]
    with open(filename1, 'r') as f1, open(filename2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    epochs = list(range(1, len(data1['metrics']['training_losses']) + 1))
    testing_accuracies1 = data1['metrics']['testing_accuracies']
    testing_accuracies2 = data2['metrics']['testing_accuracies']
    
    plt.plot(epochs, testing_accuracies1, label=f'Test Accuracy for {model1}')
    plt.plot(epochs, testing_accuracies2, label=f'Test Accuracy for {model2}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_training_times(filename1, filename2, title):
    model1 = filename1.split('/')[-1].split('.')[0]
    model2 = filename2.split('/')[-1].split('.')[0]
    with open(filename1, 'r') as f1, open(filename2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    epochs = list(range(1, len(data1['metrics']['times_per_epoch']) + 1))
    times_per_epoch1 = data1['metrics']['times_per_epoch']
    times_per_epoch2 = data2['metrics']['times_per_epoch']
    
    plt.plot(epochs, times_per_epoch1, label=f'Training Time per Epoch for {model1}')
    plt.plot(epochs, times_per_epoch2, label=f'Training Time per Epoch for {model2}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Time per Epoch (seconds)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()