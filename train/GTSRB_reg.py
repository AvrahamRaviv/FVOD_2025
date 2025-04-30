import os
import csv
import torch
from torch.utils.data import Dataset
from PIL import Image

class GTSRBDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.train = train

        if train:
            # Training data
            train_dir = os.path.join(root_dir, 'GTSRB', 'Final_Training', 'Images')
            # Iterate over class folders
            for class_id in range(43):
                class_dir = os.path.join(train_dir, f'{class_id:05d}')
                annotation_file = os.path.join(class_dir, f'GT-{class_id:05d}.csv')
                with open(annotation_file, 'r') as f:
                    reader = csv.reader(f, delimiter=';')
                    next(reader)  # skip header
                    for row in reader:
                        # Extract data
                        image_path = os.path.join(class_dir, row[0])
                        bbox = [int(row[3]), int(row[4]), int(row[5]), int(row[6])]
                        label = int(row[7])
                        self.samples.append((image_path, label, bbox))
        else:
            # Test data
            test_dir = os.path.join(root_dir, 'GTSRB', 'Final_Test', 'Images')
            annotation_file = os.path.join(root_dir, 'GT-final_test.csv')
            with open(annotation_file, 'r') as f:
                reader = csv.reader(f, delimiter=';')
                next(reader)  # skip header
                for row in reader:
                    image_path = os.path.join(test_dir, row[0])
                    bbox = [int(row[3]), int(row[4]), int(row[5]), int(row[6])]
                    label = int(row[7])
                    self.samples.append((image_path, label, bbox))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label, bbox = self.samples[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        # Convert bbox to tensor
        bbox = torch.tensor(bbox, dtype=torch.float32)
        return image, label, bbox

from torchvision.transforms import functional as F

class ResizeWithBBox:
    def __init__(self, size):
        self.size = size  # size should be a tuple (H, W)

    def __call__(self, img, bbox):
        # Get original size
        orig_width, orig_height = img.size
        img = F.resize(img, self.size)
        # Calculate scale factors
        scale_x = self.size[0] / orig_width
        scale_y = self.size[1] / orig_height
        # Adjust bounding box
        bbox = bbox.clone()
        bbox[0] = bbox[0] * scale_x  # x1
        bbox[1] = bbox[1] * scale_y  # y1
        bbox[2] = bbox[2] * scale_x  # x2
        bbox[3] = bbox[3] * scale_y  # y2
        return img, bbox

# Define transforms
class TransformWithBBox:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bbox):
        for t in self.transforms:
            img, bbox = t(img, bbox)
        return img, bbox

# Compose transforms
transform = TransformWithBBox([
    ResizeWithBBox((32, 32)),
    lambda img, bbox: (transforms.ToTensor()(img), bbox)
])

# Create datasets
train_dataset = GTSRBDataset(root_dir='data', train=True, transform=transform)
test_dataset = GTSRBDataset(root_dir='data', train=False, transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=64, shuffle=False, num_workers=4)

class NeuralNetwork_GTSRB(nn.Module):
    '''
    Updated convolutional model with classification and bounding box regression heads
    '''
    def __init__(self):
        super(NeuralNetwork_GTSRB, self).__init__()
        padding = 1

        # Adjusted input channels to 3 for RGB images
        self.conv0 = nn.Conv2d(3, 16, 3, padding=padding)
        self.pool0 = nn.MaxPool2d(2, stride=2)
        self.conv1 = nn.Conv2d(16, 16, 3, padding=padding)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 8 * 8, 256)

        # Classification head
        self.classifier = nn.Linear(256, 43)

        # Regression head for bounding boxes
        self.regressor = nn.Linear(256, 4)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = self.pool0(x)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))

        # Outputs
        class_logits = self.classifier(x)
        bbox_preds = self.regressor(x)

        return class_logits, bbox_preds


import torch.nn as nn

if __name__ == '__main__':
    # Initialize the model
    model = NeuralNetwork_GTSRB()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define loss functions and optimizer
    classification_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_class_loss = 0.0
        running_reg_loss = 0.0
        running_corrects = 0

        for inputs, labels, bboxes in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            class_outputs, bbox_outputs = model(inputs)
            _, preds = torch.max(class_outputs, 1)

            # Compute losses
            class_loss = classification_criterion(class_outputs, labels)
            reg_loss = regression_criterion(bbox_outputs, bboxes)
            loss = class_loss + reg_loss  # Total loss

            # Backward + optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_class_loss += class_loss.item() * inputs.size(0)
            running_reg_loss += reg_loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_class_loss = running_class_loss / len(train_dataset)
        epoch_reg_loss = running_reg_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'Class Loss: {epoch_class_loss:.4f} - '
              f'Reg Loss: {epoch_reg_loss:.4f} - '
              f'Acc: {epoch_acc:.4f}')

        # Evaluate on the test set
        model.eval()
        test_corrects = 0
        test_reg_loss = 0.0

        with torch.no_grad():
            for inputs, labels, bboxes in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                bboxes = bboxes.to(device)

                class_outputs, bbox_outputs = model(inputs)
                _, preds = torch.max(class_outputs, 1)

                # Compute losses
                class_loss = classification_criterion(class_outputs, labels)
                reg_loss = regression_criterion(bbox_outputs, bboxes)

                test_corrects += torch.sum(preds == labels.data)
                test_reg_loss += reg_loss.item() * inputs.size(0)

        test_acc = test_corrects.double() / len(test_dataset)
        test_reg_loss = test_reg_loss / len(test_dataset)

        print(f'Test Acc: {test_acc:.4f} - Test Reg Loss: {test_reg_loss:.4f}')

        # Save the model if it has the best accuracy so far
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print('Best model saved.')

    print(f'Best Test Acc: {best_accuracy:.4f}')

