from torchvision.models import resnet50
from torchvision.models import mobilenet_v2
from torch import nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # first: CONV => RELU => CONV => RELU => POOL set
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)

        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.norm1_2 = nn.BatchNorm2d(64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # second: CONV => RELU => CONV => RELU => POOL set
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        #

        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.norm2_2 = nn.BatchNorm2d(128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # third : CONV => RELU => CONV => RELU => POOL set
        self.conv3_1 = nn.Conv2d(128, 128, 3, padding=2)
        self.norm3_1 = nn.BatchNorm2d(128)
        # Relu
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=2)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # fully connected (single) to RELU

        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.normfc_1 = nn.BatchNorm1d(128)
        self.dropoutfc_1 = nn.Dropout2d(0.50)

        self.fc2 = nn.Linear(128, 153)

    def forward(self, x):
        out = F.relu(self.conv1_1(x))
        out = F.relu(self.norm1_2(self.conv1_2(out)))
        out = self.pool1(out)

        out = F.relu(self.conv2_1(out))
        out = F.relu(self.norm2_2(self.conv2_2(out)))
        out = self.pool2(out)

        out = F.relu(self.norm3_1(self.conv3_1(out)))
        out = F.relu(self.conv3_2(out))
        out = self.pool3(out)

        # flatten
        out = out.view(-1, 128 * 7 * 7)

        out = F.relu(self.normfc_1(self.fc1(out)))
        out = self.dropoutfc_1(out)

        out = self.fc2(out)

        return out


model = MyModel()
