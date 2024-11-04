import torch
import torch.nn as nn

# Define a simple linear classifier model
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)