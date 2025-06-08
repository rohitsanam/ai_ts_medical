import torch
import torch.nn as nn
import torchvision.models as models

class MedicalDenseNet(nn.Module):
    def __init__(self, num_classes):
        super(MedicalDenseNet, self).__init__()
        self.model = models.densenet121(pretrained=False)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)