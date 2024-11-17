import os
import subprocess
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from utils import log_metrics, select_random_images, evaluate_model



import os
import subprocess
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from utils import log_metrics, select_random_images, evaluate_model

# Create static directory and ensure index.html exists
def setup_static_directory():
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    os.makedirs(static_dir, exist_ok=True)
    
    # Check if index.html exists in static directory, if not, create it
    index_path = os.path.join(static_dir, 'index.html')
    if not os.path.exists(index_path):
        print("[WARNING] index.html not found in static directory. Please ensure it exists.")



def start_server():
    try:
        print("[INFO] Starting server...")
        server_process = subprocess.Popen(
            ["python", "server.py"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        time.sleep(2)
        print("[INFO] Server started at http://localhost:8080")
        return server_process
    except Exception as e:
        print(f"[ERROR] Failed to start server: {e}")
        return None

def stop_server(server_process):
    if server_process:
        print("[INFO] Stopping server...")
        server_process.terminate()
        server_process.wait()
        print("[INFO] Server stopped.")

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(256 * 7 * 7, 512)
            self.fc2 = nn.Linear(512, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = nn.MaxPool2d(2)(x)
            x = torch.relu(self.conv3(x))
            x = torch.relu(self.conv4(x))
            x = nn.MaxPool2d(2)(x)
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Load MNIST data
    batch_size = 512
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize metrics storage
    logs = {
        "loss": [],
        "accuracy": [],
        "epochs": [],
        "batches": [],
        "timestamps": []
    }
    
    epochs = 1
    update_frequency = 10  # Update metrics every 50 batches
    start_time = time.time()
    
    total_batches = len(train_loader)
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0
        running_correct = 0
        running_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            running_total += targets.size(0)
            running_correct += predicted.eq(targets).sum().item()
            
            # Log metrics at specified frequency or at end of epoch
            if (batch_idx + 1) % update_frequency == 0 or (batch_idx + 1) == total_batches:
                current_loss = running_loss / (batch_idx + 1)
                current_accuracy = 100. * running_correct / running_total
                current_time = time.time() - start_time
                
                # Store all metrics
                logs["loss"].append(current_loss)
                logs["accuracy"].append(current_accuracy)
                logs["epochs"].append(epoch)
                logs["batches"].append(batch_idx + 1)
                logs["timestamps"].append(current_time)
                
                # Send to frontend
                log_metrics({
                    "loss": logs["loss"],
                    "accuracy": logs["accuracy"],
                    "epochs": logs["epochs"],
                    "batches": logs["batches"],
                    "timestamps": logs["timestamps"]
                })
                
                print(f"[Epoch {epoch}, Batch {batch_idx+1}/{total_batches}] "
                      f"Loss: {current_loss:.4f}, "
                      f"Accuracy: {current_accuracy:.2f}%")
        
        print(f"[Epoch {epoch} Complete] "
              f"Loss: {current_loss:.4f}, "
              f"Accuracy: {current_accuracy:.2f}%")

if __name__ == "__main__":
    setup_static_directory()
    server_process = start_server()
    
    try:
        train_model()
    except KeyboardInterrupt:
        print("[INFO] Training interrupted by user.")
    finally:
        stop_server(server_process)




