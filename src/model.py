# model.py

from torchvision.models import resnet50
from torchvision.models import mobilenet_v2
from torch import nn

# model = resnet50(pretrained=True)
pretrained = mobilenet_v2(pretrained=True)


class MyMobileNet(nn.Module):
    def __init__(self, my_pretrained_model):
        super(MyMobileNet, self).__init__()
        self.pretrained = my_pretrained_model
        self.my_new_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1000, 200),
            nn.ReLU(),
            nn.Linear(200, 2)
        )

    def forward(self, x):
        x = self.pretrained(x)
        x = self.my_new_layers(x)
        return x


model = MyMobileNet(my_pretrained_model=pretrained)
