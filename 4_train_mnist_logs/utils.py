import json
import random
import torch
import os
from torchvision.utils import save_image

def log_metrics(logs, filepath="static/logs.json"):
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    # Save logs with absolute path
    abs_filepath = os.path.join(os.path.dirname(__file__), filepath)
    with open(abs_filepath, "w") as f:
        json.dump(logs, f)
    print(f"[INFO] Metrics logged to {abs_filepath}")

def select_random_images(dataset, num_images=10):
    indices = random.sample(range(len(dataset)), num_images)
    return [dataset[i] for i in indices]

def evaluate_model(model, data_loader, device, save_dir="results"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        images, labels = next(iter(data_loader))
        images, labels = images[:10].to(device), labels[:10]
        outputs = model(images)
        _, predictions = outputs.max(1)
        for i, (img, pred, label) in enumerate(zip(images, predictions, labels)):
            save_image(img.cpu(), f"{save_dir}/img_{i}_pred_{pred}_label_{label}.png")