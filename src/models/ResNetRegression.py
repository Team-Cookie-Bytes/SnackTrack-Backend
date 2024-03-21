from torch import nn
from torchvision import models


class ResNetRegression(nn.Module):
    def __init__(self):
        super(ResNetRegression, self).__init__()
        # Load a pre-trained ResNet
        self.resnet = models.resnet18(pretrained=True)
        # Modify the fully connected layer to output a single value
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)  # Predicting a single value

    def forward(self, x):
        x = self.resnet(x)
        return x
