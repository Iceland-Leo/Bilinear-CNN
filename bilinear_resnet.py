import torch
import torchvision
import torch.nn as nn
import torch.nn.functional


class BCNN(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(BCNN, self).__init__()
        features = torchvision.models.resnet34(pretrained=pretrained)
        # Remove the pooling layer and full connection layer
        self.conv = nn.Sequential(*list(features.children())[:-2])
        self.fc = nn.Linear(512 * 512, num_classes)
        self.softmax = nn.Softmax()

        if pretrained:
            for parameter in self.conv.parameters():
                parameter.requires_grad = False
            nn.init.kaiming_normal_(self.fc.weight.data)
            nn.init.constant_(self.fc.bias, val=0)

    def forward(self, input):
        features = self.conv(input)
        # Cross product operation
        features = features.view(features.size(0), 512, 14 * 14)
        features_T = torch.transpose(features, 1, 2)
        features = torch.bmm(features, features_T) / (14 * 14)
        features = features.view(features.size(0), 512 * 512)
        # The signed square root
        features = torch.sign(features) * torch.sqrt(torch.abs(features) + 1e-12)
        # L2 regularization
        features = torch.nn.functional.normalize(features)

        out = self.fc(features)
        softmax = self.softmax(out)
        return out, softmax
