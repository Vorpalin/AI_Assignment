# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 14:08:22 2025

@author: MIALON alexis

Link to the cancer dataset: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
"""
#%% [1] import library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim # for optimisation
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb  # Import W&B

#%% [2] Part 1

# Check GPU availability
print("CUDA Available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(20, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

model = SimpleModel().to(device)
print(model)  # Print model summary-like info

# Generate dummy data
data = np.random.rand(10, 20).astype(np.float32)  # 10 samples, 20 features each
data_torch = torch.from_numpy(data).to(device)

# Forward pass
with torch.no_grad():  # Disable gradient calculations for this test
    predictions = model(data_torch)

print("Predictions:", predictions)

#%% [3] Preprocessing


def load(filename: str, separator: str=",") -> pd.DataFrame:
    """
    Load a dateset from a csv file

    Parameters
    ----------
    filename : str
        The name of the dataset (a csv file)
    separator : str, optional
        The separator use in the csv file. The default is ",".

    Returns
    -------
    _ : pd.DataFrame
        A panda dataset

    """
    return pd.read_csv(filename, sep=separator) # load the dataset

#%% [3] Part 2

class CSVDataset(torch.utils.data.Dataset):
    def __init__(self, csv, transform=None):
        self.data = load(csv)
        self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, ind):
        
        r = self.data.iloc[ind].values
        image  = torch.tensor(r[:-1], dtype=torch.float32).view(28, 28)  # reshaped image at the good format
        label = torch.tensor(int(r[-1]), dtype=torch.long) # get the label
        
        
        image = transforms.ToPILImage()(image) # convert the image
        
        if self.transform:
            image = self.transform(image)

        return image, label
    
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=12)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout1 = nn.Dropout(0.25)  
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  
        self.pool2 = nn.MaxPool2d(2, 2)  
        self.dropout2 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(10368, 128)
        self.fc2 = nn.Linear(128, 10)  
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)

        x = self.pool2(F.relu(self.conv3(x)))
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
#%% [4] Main code

if __name__ == '__main__':
    
    # login to wandb
    k = input("What is your wandb key: ")
    wandb.login(key=k)
    wandb.init(project="mnist-cnn", name="experiment_1")
    # optimization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # transformer for preprocessing and data augmentation
    transform_augmentation = transforms.Compose([
        #transforms.Grayscale(num_output_channels=3),  # Convert 1-channel to 3-channel
        transforms.RandomRotation(20),
        transforms.Resize((28, 28)),
        transforms.RandomResizedCrop(28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_classic = transforms.Compose([
       # transforms.Grayscale(num_output_channels=3),  # Convert 1-channel to 3-channel
        transforms.Resize((28, 28)),  # Ensure size matches training
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    transform_test = transforms.Compose([
       # transforms.Grayscale(num_output_channels=3),  # Convert 1-channel to 3-channel
        transforms.RandomRotation(15),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
   
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_classic)
    train_dataset_2 = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_augmentation)
    test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    
    train_data = torch.utils.data.ConcatDataset([train_dataset, train_dataset_2])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)  # shuffle=True
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)  # no shuffle for testing
   
    # put the CNN to the cpu
    cnn = CNN().to(device)

    # optimisation
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    training = ""
    start_epoch = 1
    
    #
    wandb.watch(model, log="all")
    # ask the number of epochs
    try:
        nb_epoch = int(input("How many epoch do you want to execute: "))
    except:
        nb_epoch = 20
    
    # training
    all_epoch = []
    epoch_l = []
    test_epoch = []
    best_val_loss = float('inf')  
    patience = 5  
    epochs_without_improvement = 0  
    # training with early stopping
    for epoch in range(start_epoch,start_epoch+nb_epoch):
        model.train()
        running_loss = 0.0
        c = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs and the label to the gpu
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward
            outputs = cnn(inputs)
                        
            # compute loss
            loss = criterion(outputs, labels)
            
            # backward
            loss.backward()
            _, predictions = torch.max(outputs,1)
            c +=  (predictions == labels).sum().item()
            # optimize
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            
        print(f'[Epoch {epoch}] loss: {running_loss / len(train_data):.3f}')
        # store loss values
        all_epoch.append(running_loss / len(train_data))
        epoch_l.append(epoch)
        
        # create dictionary to get accuracy for each digit
        correct_pred = {i: 0 for i in range(10)}
        total_pred = {i: 0 for i in range(10)}
        correct = 0
        total = 0
        running_loss = 0.0
        # test the data
        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = cnn(images)
                _, predictions = torch.max(outputs,1)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                # if the prediction is good we add one to the counter
                if torch.tensor([labels]) == predictions:
                    correct_pred[labels.item()] += 1
                    correct += 1
                total_pred[labels.item()] += 1
                total += 1


        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname} is {accuracy:.1f} %')
            
        print(f'Accuracy: {correct*100/total:.1f}%')
        print(f'Loss: {running_loss/len(test_loader)}')
        test_epoch.append(running_loss/len(test_loader))
        wandb.log({
        "epoch": epoch ,
        "train_loss": c/len(train_loader),
        "train_accuracy": correct/total,
        "test_loss": all_epoch[-1],
        "test_accuracy": test_epoch[-1]
    })
        scheduler.step()
        # early stopping
        if running_loss /len(test_loader) < best_val_loss:
            best_val_loss = running_loss /len(test_loader)
            torch.save(cnn.model.state_dict(), "best_model.pth""model.pth")  # Save model
            print("Best model saved")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement == patience:
                break
    print('Finished Training')
    
    # graph of the loss during training
    plt.plot(epoch_l, all_epoch, label="Training Loss", marker='o')
    plt.plot(epoch_l, test_epoch, label="Validation Loss", marker='s')
    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    wandb.finish()
    
