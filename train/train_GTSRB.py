import torch
import torch.nn.functional as F
from torch import nn
import torchvision
import torchvision.transforms as transforms
import os
import csv
from tqdm import tqdm

class NeuralNetwork_GTSRB(nn.Module):
    '''
    Updated convolutional model
    '''
    def __init__(self, head='class'):
        super(NeuralNetwork_GTSRB, self).__init__()
        self.head = head
        padding = 1

        # Adjusted input channels to 3 for RGB images
        self.conv0 = nn.Conv2d(3, 16, 3, padding=padding)
        self.pool0 = nn.MaxPool2d(2, stride=2)
        self.conv1 = nn.Conv2d(16, 16, 3, padding=padding)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.flatten = nn.Flatten()

        # After resizing images to 32x32 and applying the conv and pool layers,
        # the output size will be 16 * 8 * 8 = 1024
        self.fc1 = nn.Linear(16 * 8 * 8, 256)
        if self.head == 'class' or self.head == 'both':
            # Updated number of classes to 43 for GTSRB
            self.linear = nn.Linear(256, 43)
        if self.head == 'reg' or self.head == 'both':
            self.linear_all = nn.Linear(256, 4)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = self.pool0(x)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        if self.head == 'class':
            logits = self.linear(x)
        elif self.head == 'reg':
            logits = self.linear_all(x)
        else:
            logits = (self.linear(x), self.linear_all(x))
        return logits

if __name__ == '__main__':
    # Set the model to classification mode
    model = NeuralNetwork_GTSRB(head='class')

    # Define transforms: resize images and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((90, 90)),  # Resize images to 32x32
        transforms.ToTensor()
    ])

    # Load datasets with the new transform
    train_data = torchvision.datasets.GTSRB(
        root='data', split="train", download=True, transform=transform)

    # train labels and bboxes
    # load bbox data from train.csv
    # the format of each line is: Width	Height	Roi.X1	Roi.Y1	Roi.X2	Roi.Y2
    labels = []
    bboxes = []
    # Load annotation file
    with open('Train.csv', "r") as csvfile:
        rows = csv.reader(csvfile)
        next(rows)
        # Loop rows
        for row in tqdm(rows):
            # Obtain each data from the csv
            (w, h, startX, startY, endX, endY, label, relativeFilePath) = row

            # scale the bounding box coordinates relative to the spatial
            # dimensions of the input image
            startX = float(startX) / float(w) * 90
            startY = float(startY) / float(h) * 90
            endX = float(endX) / float(w) * 90
            endY = float(endY) / float(h) * 90

            # update our list of class labels, bounding boxes, and
            # image paths
            labels.append(int(label))
            bboxes.append((startX, startY, endX, endY))

    train_labels = labels
    train_bboxes = bboxes

    # verify that labels are equal to train_data lables
    for i in range(len(train_data)):
        assert train_labels[i] == train_data[i][1]

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=64, shuffle=True)

    test_data = torchvision.datasets.GTSRB(
        root='data', split="test", download=True, transform=transform)
    # load bbox data from test.csv
    # the format of each line is: Width	Height	Roi.X1	Roi.Y1	Roi.X2	Roi.Y2
    labels = []
    bboxes = []
    # Load annotation file
    with open('Test.csv', "r") as csvfile:
        rows = csv.reader(csvfile)
        next(rows)
        # Loop rows
        for row in tqdm(rows):
            # Obtain each data from the csv
            (w, h, startX, startY, endX, endY, label, relativeFilePath) = row

            # scale the bounding box coordinates relative to the spatial
            # dimensions of the input image
            startX = float(startX) / float(w)
            startY = float(startY) / float(h)
            endX = float(endX) / float(w)
            endY = float(endY) / float(h)

            # update our list of class labels, bounding boxes, and
            # image paths
            labels.append(int(label))
            bboxes.append((startX, startY, endX, endY))
    test_labels = labels
    test_bboxes = bboxes

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Number of epochs
    num_epochs = 10  # Replace 10 with your desired number of epochs

    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward + optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_data)
        epoch_acc = running_corrects.double() / len(train_data)

        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Evaluate on the test set
        model.eval()
        test_corrects = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                test_corrects += torch.sum(preds == labels.data)

        test_acc = test_corrects.double() / len(test_data)
        print(f'Test Acc: {test_acc:.4f}')

        # Save the model if it has the best accuracy so far
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print('Best model saved.')

    print(f'Best Test Acc: {best_accuracy:.4f}')
