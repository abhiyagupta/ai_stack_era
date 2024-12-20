import torch
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
#from model_architecture.model import Net
#from model_architecture.model_5_red_param import Net
from model_architecture.model_6_red_param import Net
from data_modules.mnist_data_prep import MnistDataModule
import torch.nn.functional as F

def save_inference_image(image, prediction, confidence, save_dir, idx):
    """Save the image with its prediction and confidence score"""
    plt.figure(figsize=(3, 3))
    plt.imshow(image.squeeze().cpu(), cmap='gray')
    plt.axis('off')
    plt.title(f'Pred: {prediction}\nConf: {confidence:.2f}%')
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'inference_{idx}_pred_{prediction}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def infer_random_samples(model, data_loader, save_dir, num_samples=10, device='cpu'):
    """Perform inference on random samples and save visualizations"""
    model.eval()
    predictions = []
    confidences = []
    
    # Convert dataloader to list for random sampling
    all_data = [(data, target) for data, target in data_loader]
    
    # Randomly sample indices
    total_samples = len(all_data)
    random_indices = random.sample(range(total_samples), min(num_samples, total_samples))
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(random_indices):
            data, _ = all_data[sample_idx]
            data = data.to(device)
            
            # Get model output
            output = model(data)
            probs = F.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)
            
            # Save prediction and confidence
            pred_val = pred.item()
            conf_val = conf.item() * 100
            
            # Save visualization
            save_inference_image(data[0], pred_val, conf_val, 
                               os.path.join(save_dir, 'random_samples'), idx)
            
            predictions.append(pred_val)
            confidences.append(conf_val)
            
            print(f'Sample {idx+1}/10: Prediction={pred_val}, Confidence={conf_val:.2f}%')
    
    return predictions, confidences

def main():
    save_dir = "./inference_results"
    os.makedirs(save_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model_path = "./models/best_model.pt"
    model = Net().to(device)
    model.load_state_dict(torch.load(model_path))
    print(f"Loaded model from {model_path}")
    
    print("\nModel Architecture:")
    model.display_architecture()

    print("\nLoading dataset...")
    data_module = MnistDataModule()
    data_module.setup()
    inference_loader = data_module.get_inference_dataloader()

    # Perform inference on random samples
    print("\nPerforming inference on 10 random samples...")
    predictions, confidences = infer_random_samples(model, inference_loader, save_dir, num_samples=10, device=device)
    
    # Save inference statistics
    stats_file = os.path.join(save_dir, 'random_samples_stats.txt')
    with open(stats_file, 'w') as f:
        f.write("Random Samples Inference Statistics\n")
        f.write("================================\n\n")
        for i in range(len(predictions)):
            f.write(f"Sample {i}: Predicted {predictions[i]} with confidence {confidences[i]:.2f}%\n")
    
    print(f"\nInference completed. Results saved in {save_dir}")
    print(f"- Random Sample Visualizations: {os.path.join(save_dir, 'random_samples')}")
    print(f"- Statistics: {stats_file}")

if __name__ == "__main__":
    main()