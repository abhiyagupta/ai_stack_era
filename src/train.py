import os
import torch
import torch.optim as optim
from tqdm import tqdm
#from model_architecture.model import Net
#from model_architecture.model_5_red_param import Net
from model_architecture.model_6_red_param import Net
from data_modules.mnist_data_prep import MnistDataModule
from eval import evaluate
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_epoch(model, train_loader, optimizer, epoch, device='cpu'):
    model.train()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

def main():
    # Hyperparameters
    batch_size = 64
    lr = 0.01
    momentum = 0.9       #0.5
    weight_decay = 1e-4
    epochs = 2 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = "./models"
    os.makedirs(save_dir, exist_ok=True)

    print("Loading datasets...")
    data_module = MnistDataModule(batch_size=batch_size)
    data_module.setup()
    
    train_loader = data_module.get_train_dataloader()  # For training
    val_loader = data_module.get_val_dataloader()

    print(f"Using device: {device}")
    print("Initializing model...")
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # adding weight decay -L2 regulaization
    #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True, threshold=0.0001, threshold_mode='rel',
                                                       cooldown=0, min_lr=0, eps=1e-08)


    print("\nModel Architecture:")
    model.display_architecture()



    best_accuracy = 0.0
    
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        train_epoch(model, train_loader, optimizer, epoch, device)
       

        # Validate
        val_accuracy = evaluate(model, val_loader, device)
        print(f"\nEpoch {epoch}: Validation Accuracy: {val_accuracy:.2f}%")


        # Update scheduler based on validation accuracy
        scheduler.step(val_accuracy)


        #save the best model         
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
            print(f"Saved best model with accuracy: {best_accuracy:.2f}%")

if __name__ == "__main__":
    main()