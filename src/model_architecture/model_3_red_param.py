import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Reduced filters further
        self.conv1 = nn.Conv2d(1, 8, 3, 1)  # Reduced from 16 to 8 filters
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)  # Reduced from 32 to 16 filters
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # Greatly reduced dense layer sizes
        self.fc1 = nn.Linear(2304, 32)  # Reduced from 4608, 64 to 2304, 32
        self.bn3 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def display_architecture(self):
        """Display the model architecture in a readable format"""
        print("\nModel Architecture Summary:")
        print("=" * 50)
        summary(self, input_size=(1, 1, 28, 28), 
               verbose=2,
               col_names=["input_size", "output_size", "num_params", "kernel_size"],
               device='cpu')
        
        print("\nLayer Details:")
        print("=" * 50)
        for name, module in self.named_children():
            print(f"{name}: {module}")

if __name__ == "__main__":
    model = Net()
    model.display_architecture()