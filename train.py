import os
import boto3 
import torch        
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import Callback
import yaml
from model import ResNetLightningModule
from data_module import ImageNetDataModule
from typing import Optional
import tempfile
import random
import numpy as np 

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Optional: for absolute reproducibility

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def load_config(config_path: str = 'config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)



def load_latest_checkpoint(bucket_name: str, s3_prefix: str) -> Optional[str]:
    """Load the latest checkpoint from S3 if it exists."""
    s3_client = boto3.client('s3')
    
    try:
        # List all checkpoints in the S3 prefix
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=f"{s3_prefix}/checkpoints/"
        )
        
        if 'Contents' not in response:
            return None
            
        # Find the latest checkpoint
        checkpoints = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.ckpt')]
        if not checkpoints:
            return None
            
        latest_checkpoint = max(checkpoints, key=lambda x: s3_client.head_object(Bucket=bucket_name, Key=x)['LastModified'])
        
        # Download to a temporary file
        temp_dir = tempfile.mkdtemp()
        local_path = os.path.join(temp_dir, os.path.basename(latest_checkpoint))
        s3_client.download_file(bucket_name, latest_checkpoint, local_path)
        
        return local_path
    except Exception as e:
        print(f"Error loading checkpoint from S3: {e}")
        return None




class S3ModelCheckpoint(ModelCheckpoint):
    def __init__(self, bucket_name: str, s3_prefix: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bucket_name = bucket_name
        self.s3_prefix = s3_prefix
        self.s3_client = boto3.client('s3')
    
    def _save_model(self, trainer, filepath: str):
        try:
            # First save locally
            super()._save_model(trainer, filepath)
            
            # Then upload to S3
            
            s3_path = f"{self.s3_prefix}/{os.path.basename(filepath)}" # Remove hardcoded /checkpoints/
            #s3_path = f"{self.s3_prefix}/checkpoints/{os.path.basename(filepath)}"
            self.s3_client.upload_file(filepath, self.bucket_name, s3_path)
            print(f"\nCheckpoint saved to S3: s3://{self.bucket_name}/{s3_path}")
            
            # Clean up local file only after successful upload
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            print(f"\nError saving checkpoint to S3: {e}")
            # Keep local file if upload fails
            if os.path.exists(filepath):
                print(f"Local checkpoint preserved at: {filepath}")


class S3Logger(CSVLogger):
    def __init__(self, bucket_name: str, s3_prefix: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bucket_name = bucket_name
        self.s3_prefix = s3_prefix
        self.s3_client = boto3.client('s3')
    
    def save(self):
        # First save locally
        super().save()
        
        # Upload logs to S3
        local_path = self.log_dir
        for root, _, files in os.walk(local_path):
            for file in files:
                local_file = os.path.join(root, file)
                s3_path = f"{self.s3_prefix}/logs/{file}"
                self.s3_client.upload_file(local_file, self.bucket_name, s3_path)
        print(f"\nLogs saved to S3: s3://{self.bucket_name}/{self.s3_prefix}/logs/")

#checks if 2 models have reached 70% acc --training will stop 
class AccuracyThresholdCallback(Callback):
    def __init__(self, threshold: float = 70.0, num_models: int = 2): 
        super().__init__()
        self.threshold = threshold
        self.num_models = num_models
        self.top_accuracies = []  # Remove type hint for compatibility
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # Get current validation accuracy
        current_acc = trainer.callback_metrics.get('val_acc', 0) * 100  # Convert to percentage
        
        # Update top accuracies list
        self.top_accuracies.append(current_acc)
        self.top_accuracies.sort(reverse=True)  # Sort in descending order
        self.top_accuracies = self.top_accuracies[:self.num_models]  # Keep only top N
        
        # Check if we have enough models and all meet threshold
        if len(self.top_accuracies) == self.num_models:
            if all(acc >= self.threshold for acc in self.top_accuracies):
                print(f"\nStopping training as top {self.num_models} models have reached "
                      f"accuracy threshold of {self.threshold}%")
                print(f"Top accuracies: {self.top_accuracies}")
                trainer.should_stop = True




class EpochMetricsCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        val_acc = metrics.get('val_acc', 0) * 100  # Convert to percentage
        train_acc = metrics.get('train_acc', 0) * 100  # Convert to percentage
        
        print(f"\nEpoch {trainer.current_epoch}:")
        print(f"Training Accuracy: {train_acc:.2f}%")
        print(f"Validation Accuracy: {val_acc:.2f}%")



def main():
    # Load configuration
    config = load_config()

    # Set random seed before doing anything else
    set_seed(config['training']['seed'])
    
    # Set seed for PyTorch Lightning
    pl.seed_everything(config['training']['seed'], workers=True)
    
    if not config['s3']['bucket_name']:
        raise ValueError("S3 bucket name must be specified in config.yaml")



    
    # Create temporary directory for checkpoints
    temp_checkpoint_dir = tempfile.mkdtemp()
    
    # Initialize S3 checkpoint callback
    checkpoint_callback = S3ModelCheckpoint(
        bucket_name=config['s3']['bucket_name'],
        s3_prefix=config['s3']['checkpoint_prefix'],
        dirpath=temp_checkpoint_dir,
        filename='model-{epoch:02d}-{val_acc:.4f}',
        save_top_k=2,  #3 # Saves only the checkpoint with highest validation accuracy
        monitor='val_acc',
        mode='max',
        save_last=True,  # This saves the last checkpoint separately
        every_n_epochs=1,
        verbose=True  # Add verbose output
    )
    
    # Initialize accuracy threshold callback
    accuracy_callback = AccuracyThresholdCallback(
        threshold=70.0,  # 70% accuracy threshold
        num_models=1     # Check top 3 models
    )
    

    epoch_metrics_callback = EpochMetricsCallback()


    # Initialize S3 logger
    logger = S3Logger(
        bucket_name=config['s3']['bucket_name'],
        s3_prefix=config['s3']['logs_prefix'],           #'imagenet-training'
        save_dir=temp_checkpoint_dir,
        name='imagenet_logs'
    )
    
    # Initialize data module and model
    data_module = ImageNetDataModule(config)
    
    # Try to load the latest checkpoint from S3
    # resume_checkpoint = load_latest_checkpoint(
    #     config['s3']['bucket_name'],
    #     'imagenet-training'
    # )
    resume_checkpoint = load_latest_checkpoint(
    config['s3']['bucket_name'],
    config['s3']['checkpoint_prefix']
    )



    # Initialize model
    model = ResNetLightningModule(config)
    
    # Initialize trainer with updated configuration
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator="gpu",
        devices="auto",
        precision=16,
        callbacks=[checkpoint_callback, accuracy_callback,epoch_metrics_callback],  # Added accuracy callback
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=True,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        sync_batchnorm=True,
        enable_checkpointing=True
        #resume_from_checkpoint=resume_checkpoint
    )
    
    # Train model
    trainer.fit(model, data_module,ckpt_path=resume_checkpoint)
    
    # Clean up temporary directory
    import shutil
    shutil.rmtree(temp_checkpoint_dir)

if __name__ == "__main__":
    main()



