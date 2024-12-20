import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def display_architecture(self):
        """Display the model architecture in a readable format"""
        # Create a sample input tensor (batch_size, channels, height, width)
        sample_input = torch.zeros(1, 1, 28, 28)
        print("\nModel Architecture Summary:")
        print("=" * 50)
        # Use torchinfo to display detailed architecture
        summary(self, input_size=(1, 1, 28, 28), 
               verbose=2,
               col_names=["input_size", "output_size", "num_params", "kernel_size"],
               device='cpu')
        
        # Additional architecture details
        print("\nLayer Details:")
        print("=" * 50)
        for name, module in self.named_children():
            print(f"{name}: {module}")

if __name__ == "__main__":
    # Display architecture when run directly
    model = Net()
    model.display_architecture()