import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Adjusted number of filters
        self.conv1 = nn.Conv2d(1, 6, 3, 1)  # Reduced to 6 filters
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 12, 3, 1)  # Reduced to 12 filters
        self.bn2 = nn.BatchNorm2d(12)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Efficient pooling structure
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate the size after convolutions and pooling:
        # 28x28 -> 26x26 (conv1) -> 13x13 (pool) -> 11x11 (conv2) -> 5x5 (pool)
        # Final size: 5 x 5 x 12 = 300
        self.fc1 = nn.Linear(300, 48)  # Balanced size
        self.bn3 = nn.BatchNorm1d(48)
        self.fc2 = nn.Linear(48, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
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


# ========================================================================


# Convolutional Layers:
# Conv1: (1 x 3 x 3 +1) x 6=60
# Conv2: (6 x3 x 3 +1) x12 =660

# Fully Connected Layers:
# FC1: 300 X 48 +  48 =14,448
# FC2: 48 x 10 + 10 = 490

# Batch Norm Layers:
# BN1: 6 x 2 =12
# BN2: 12 x 2 = 24
# BN3: 48 x 2 = 96

# Total Parameters: 
# 60 +660 + 14,448 + 490 +12 + 24 + 96=15,790 
# This architecture satisfies the constraint of having total parameters less than  20,000.