import torch
from torch import nn
import pickle

# load LARD model and test set, and measure its mean IoU
class Neural_network_LARD(nn.Module):
    def __init__(self):
        super(Neural_network_LARD, self).__init__()
        self.conv0 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1)
        self.flatten = nn.Flatten()  # 131072
        self.linear7 = nn.Linear(131072, 128)
        self.linear9 = nn.Linear(128, 128)
        self.linear11 = nn.Linear(128, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear7(x)
        x = self.relu(x)
        x = self.linear9(x)
        x = self.relu(x)
        x = self.linear11(x)
        return (x)


model = Neural_network_LARD()
# load 'LARD_weights.pt' using torch.jit.load
_sd = torch.load('LARD_weights.pt', map_location=torch.device('cpu')).state_dict()
sd = {'conv' + k if k.startswith(('0', '2', '4')) else 'linear' + k if k.startswith(('7', '9', '11')) else k: v for k, v
      in _sd.items()}
model.load_state_dict(sd)

test_pkl = "LARD_test.pkl"
with open(test_pkl, 'rb') as f:
    testingData = pickle.load(f)

X = testingData['x_train'] / 255
labels = testingData['y_train'] * 256
# take first 10 samples for testing
X = torch.tensor(X)
labels = torch.tensor(labels)

# predict bounding boxes
mean_IoU = 0
for i in range(len(X)):
    img = X[i].unsqueeze(0)
    label = labels[i].unsqueeze(0)
    pred = model(img.float())
    pred = pred.detach().numpy()
    pred = pred
    pred = pred[0]
    # pred = pred.astype(int)
    label = label.numpy()
    label = label[0]
    # label = label.astype(int)
    # calculate IoU
    xA = max(pred[0], label[0])
    yA = max(pred[1], label[1])
    xB = min(pred[2], label[2])
    yB = min(pred[3], label[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (pred[2] - pred[0] + 1) * (pred[3] - pred[1] + 1)
    boxBArea = (label[2] - label[0] + 1) * (label[3] - label[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    mean_IoU += iou

mean_IoU = mean_IoU / len(X)
