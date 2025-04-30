import torch
import torch.nn.functional as F
from torch import nn
from custom_from_mnist import CustomMnistDataset_OL
import torchvision
import torchvision.transforms as transforms


class NeuralNetwork_OL_v2(nn.Module):
    '''
    New convolutional model (v2)
    '''

    def __init__(self):
        super(NeuralNetwork_OL_v2, self).__init__()
        seed = 0
        torch.manual_seed(seed)
        padding = 1

        self.conv0 = nn.Conv2d(1, 16, 3, padding=padding)  # 3x3 filters w/ same padding
        self.pool0 = nn.MaxPool2d(2, stride=2)
        # output shape : 15x15x16
        self.conv1 = nn.Conv2d(16, 16, 3, padding=padding)  # 3x3 filters w/ same padding

        self.pool1 = nn.MaxPool2d(2, stride=2)
        # output shape : 8x8x16
        self.flatten = nn.Flatten()
        # output shape : 1024
        # HERE CHECK RIGHT SIZE FROM FLATTEN TO LINEAR

        self.linear_relu_stack = nn.Linear(7744, 256)
        self.linear_relu_stack = nn.Linear(7744, 256)
        self.linear = nn.Linear(256, 10)
        self.linear_all = nn.Linear(256, 4)

    def forward(self, x):
        x = self.conv0(x)
        x = F.relu(self.pool0(x))
        x = self.conv1(x)
        x = F.relu(self.pool1(x))
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        x = F.relu(x)
        logits_class = self.linear(x)
        logits_reg = self.linear_all(x)
        logits = (logits_class, logits_reg)

        return logits


# main function
if __name__ == '__main__':
    max_acc = 0
    # model
    sd = torch.load('d_loc_weights.pt')
    # save sd as a dictionary (state_dict) but not in jit format
    model = NeuralNetwork_OL_v2()
    model.load_state_dict(sd, strict=False)

    # data - load MNIST dataset and preprocess using custom_from_mnist.py
    train_data = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
    train_data_custom = CustomMnistDataset_OL(train_data, test=False)
    # train loader
    train_loader = torch.utils.data.DataLoader(train_data_custom, batch_size=64, shuffle=True)
    # test
    test_data = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())
    test_data_custom = CustomMnistDataset_OL(test_data, test=True)
    test_loader = torch.utils.data.DataLoader(test_data_custom, batch_size=64, shuffle=False)

    # training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(10):
        # keep all layers in eval mode, except the last layer
        for name, param in model.named_parameters():
            param.requires_grad = False
        model.linear.requires_grad = True
        model.linear.weight.requires_grad = True
        model.linear.bias.requires_grad = True
        for i, (images, labels) in enumerate(train_loader):
            images = images.float()
            labels = labels.long()
            optimizer.zero_grad()
            outputs = model(images)[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print(f'Epoch: {epoch + 1}, Step: {i + 1}, Loss: {loss.item()}')

        # inference
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.float()
                labels = labels.long()
                outputs = model(images)[0]
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = correct / total
            print(f'Accuracy: {accuracy}')
            if accuracy > max_acc:
                max_acc = accuracy
                torch.save(model.state_dict(), 'd_loc_weights_finetuned.pt')
                print('Model saved!')


# given image with shape of [1, C, H, W] and bounding box [x_min, y_min, x_max, y_max]
# plot them using matplotlib
import matplotlib.pyplot as plt
# assume X and b are given
X = image
b = label[0]
fig, ax = plt.subplots()
X = X.numpy()
X = X[0, :, :, :]
X = X.transpose(1, 2, 0)
ax.imshow(X)
rect = plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)
plt.show()
