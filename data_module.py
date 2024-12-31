import os
import torch        
import pytorch_lightning as pl
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import multiprocessing
from typing import Optional
import random
import numpy as np 

class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.data_dir = config['data']['data_dir']
        self.batch_size = config['data']['batch_size']
        self.num_workers = config['data']['num_workers']
        self.seed = config['training']['seed']  # Add seed from config
        self.generator = torch.manual_seed(42)
        
        # If num_workers is not specified, calculate based on CPU cores
        if not self.num_workers:
            self.num_workers = max(1, multiprocessing.cpu_count() - 4)
            
        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, 
                interpolation=transforms.InterpolationMode.BILINEAR, 
                antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        self.train_dataset: Optional[datasets.ImageFolder] = None
        self.val_dataset: Optional[datasets.ImageFolder] = None

    def prepare_data(self):
        """
        Called only once and on 1 GPU.
        Used for downloading datasets, tokenize, etc.
        """
        # Check if the data directory exists
        train_dir = os.path.join(self.data_dir, 'ILSVRC/Data/CLS-LOC/train')
        val_dir = os.path.join(self.data_dir, 'ILSVRC/Data/CLS-LOC/val')
        
        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            raise FileNotFoundError(
                f"ImageNet data not found at {self.data_dir}. "
                "Please ensure the data is properly organized in the ILSVRC format."
            )

    def setup(self, stage: Optional[str] = None):
        """
        Called on every process in distributed training.
        Load datasets here.
        """
        if stage == 'fit' or stage is None:
            self.train_dataset = datasets.ImageFolder(
                root=os.path.join(self.data_dir, 'ILSVRC/Data/CLS-LOC/train'),
                transform=self.train_transforms
            )
            self.val_dataset = datasets.ImageFolder(
                root=os.path.join(self.data_dir, 'ILSVRC/Data/CLS-LOC/val'),
                transform=self.val_transforms
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            generator=self.generator,  # Add generator for reproducible shuffling
            worker_init_fn=lambda id: pl.seed_everything(self.seed + id)  # Seed workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            generator=self.generator
            
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after training/testing."""
        pass