import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import resnet50
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Any

class ResNetLightningModule(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Initialize model
        self.model = resnet50(
            pretrained=False,
            num_classes=config['data']['num_classes']
        )
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        acc = (logits.argmax(dim=1) == y).float().mean()
        
        # Log metrics to progress bar only
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.log('train_acc', acc, prog_bar=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        acc = (logits.argmax(dim=1) == y).float().mean()
        
        # Log metrics to progress bar only
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
        self.parameters(),
        lr=float(self.config['model']['learning_rate']),  # Ensure it's a float
        momentum=float(self.config['model']['momentum']),  # Ensure it's a float
        weight_decay=float(self.config['model']['weight_decay'])  # Ensure it's a float

        )
        
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=3,
                verbose=True
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1
        }
        
        return [optimizer], [scheduler]