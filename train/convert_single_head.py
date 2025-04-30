import torch
import torch.nn.functional as F
from torch import nn as nn
from custom_from_mnist import CustomMnistDataset_OL
import torchvision
import torchvision.transforms as transforms

class NeuralNetwork_OL_v2_old(nn.Module):
    '''
    New convolutional model (v2)
    '''

    def __init__(self):
        super(NeuralNetwork_OL_v2_old, self).__init__()
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
        self.new_linear = nn.Linear(256, 14)

    def forward(self, x):
        x = self.conv0(x)
        x = F.relu(self.pool0(x))
        x = self.conv1(x)
        x = F.relu(self.pool1(x))
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        x = F.relu(x)
        logits = self.new_linear(x)

        return logits

if __name__ == '__main__':
    max_acc = 0
    # model

    old_model = NeuralNetwork_OL_v2_old()
    sd = torch.load('d_loc_weights_finetuned.pt')
    old_model.load_state_dict(sd, strict=True)
    new_model = NeuralNetwork_OL_v2()
    new_model.load_state_dict(sd, strict=False)
    # load the weights from old heads to new head
    # Old heads are one for classification (256->10) and one for regression (256->4)
    new_model.new_linear.weight.data = torch.zeros_like(new_model.new_linear.weight.data)
    new_model.new_linear.weight.bias = torch.zeros_like(new_model.new_linear.bias.data)
    new_model.new_linear.weight.data[:10] = old_model.linear.weight.data
    new_model.new_linear.weight.bias[:10] = old_model.linear.bias
    new_model.new_linear.weight.data[10:] = old_model.linear_all.weight.data
    new_model.new_linear.weight.bias[10:] = old_model.linear_all.bias

    # test
    test_data = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())
    test_data_custom = CustomMnistDataset_OL(test_data, test=True)
    test_loader = torch.utils.data.DataLoader(test_data_custom, batch_size=64, shuffle=False)

    # inference
    new_model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.float()
            labels = labels.long()
            outputs = new_model(images)[:, :10]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print(f'Accuracy: {accuracy}')

    # save the new model
    scripted_model = torch.jit.script(new_model)
    torch.jit.save(scripted_model, 'new_model_scripted.pt')  # For scripted model
