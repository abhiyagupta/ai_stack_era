import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Drastically reduced filters
        self.conv1 = nn.Conv2d(1, 4, 3, 1)  # Reduced to 4 filters
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, 3, 1)  # Reduced to 8 filters
        self.bn2 = nn.BatchNorm2d(8)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Add another pooling layer to reduce dimensions
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate the size after convolutions and pooling:
        # 28x28 -> 26x26 (conv1) -> 13x13 (pool) -> 11x11 (conv2) -> 5x5 (pool)
        # Final size: 5 x 5 x 8 = 200
        self.fc1 = nn.Linear(200, 20)  # Drastically reduced
        self.bn3 = nn.BatchNorm1d(20)
        self.fc2 = nn.Linear(20, 10)

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



# Key changes to reduce parameters:

# Convolutional layers:

# First conv layer: Reduced from 8 to 4 filters
# Second conv layer: Reduced from 16 to 8 filters


# Added extra pooling:

# Now applying pooling after each conv layer
# This reduces spatial dimensions more aggressively
# Original: 28x28 -> 26x26 -> 24x24 -> 12x12
# New: 28x28 -> 26x26 -> 13x13 -> 11x11 -> 5x5


# Fully connected layers:

# Input features reduced from 2304 to 200 (5x5x8)
# Hidden layer reduced from 32 to 20 neurons
# Output stays at 10 (for classification)



# Let's calculate the approximate parameters:

# Conv1: (3×3×1×4) + 4 = 40 params
# Conv2: (3×3×4×8) + 8 = 296 params
# FC1: (200×20) + 20 = 4,020 params
# FC2: (20×10) + 10 = 210 params
# Batch Norm layers: ~42 params

# Total trainable parameters should now be well under 20,000. The biggest reduction came from:

# Using more pooling layers to reduce spatial dimensions
# Reducing number of conv filters
# Drastically reducing the size of fully connected layers