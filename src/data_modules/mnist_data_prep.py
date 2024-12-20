from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
import torch

class UnlabeledDataset(Dataset):
    """Dataset wrapper that removes labels for inference"""
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, _ = self.dataset[index]
        return data, torch.tensor(-1)  # Return -1 as dummy label

    def __len__(self):
        return len(self.dataset)

class MnistDataModule:
    def __init__(self, data_dir: str = './data', batch_size: int = 64):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
    def setup(self):
        # Load full MNIST dataset
        self.full_dataset = datasets.MNIST(
            self.data_dir, 
            train=True, 
            download=True, 
            transform=self.transform
        )
        
        # Calculate split sizes
        total_size = len(self.full_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.2 * total_size)
        inference_size = total_size - train_size - val_size  # Remaining 10%
        
        # Split dataset into train, validation, and inference
        self.train_dataset, self.val_dataset, inference_subset = \
            random_split(self.full_dataset, [train_size, val_size, inference_size])
            
        # Convert inference subset to unlabeled dataset
        self.inference_dataset = UnlabeledDataset(inference_subset)

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def get_val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
    def get_inference_dataloader(self):
        return DataLoader(
            self.inference_dataset,
            batch_size=1,  # Batch size 1 for inference
            shuffle=False,
            num_workers=4
        )