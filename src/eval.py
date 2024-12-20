import torch
from tqdm import tqdm
#from model_architecture.model import Net
#from model_architecture.model_5_red_param import Net
from model_architecture.model_6_red_param import Net
from data_modules.mnist_data_prep import MnistDataModule

def evaluate(model, data_loader, device='cpu'):
    """Evaluate the model on the provided dataset."""
    model.eval()
    correct = 0
    total = len(data_loader.dataset)
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Evaluating')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Update progress bar
            # Calculate accuracy
            accuracy = 100. * correct / total
            pbar.set_postfix({'accuracy': f'{accuracy:.2f}%'})
    
    return accuracy

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load the best model
    model_path = "./models/best_model.pt"
    model = Net().to(device)
    model.load_state_dict(torch.load(model_path))
    print(f"Loaded model from {model_path}")

    # Load the validation dataset
    print("Loading dataset...")
    data_module = MnistDataModule()
    data_module.setup()
    val_loader = data_module.get_val_dataloader()   # For validating/evalutaing

    # Evaluate the model
    val_accuracy = evaluate(model, val_loader, device)
    print(f"\nValidation Accuracy: {val_accuracy:.2f}%")

if __name__ == "__main__":
    main()