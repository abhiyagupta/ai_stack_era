# ResNet-50 Image Classification on ImageNet-1000 datset from scratch
Aim: 
- to train resnet 50 from scratch on imagenet 1000 datasets
- achieve a 70 % accuaracy in maximum 100 epochs

# Overview
- used pytorch lightening
- trained on spot instance g4dn 2xLarge
- all configurations added in config.yaml
- since spot instance, saved model checkpoints to resume training
- added seed for reproduction to resume training
- model checkpoints and logs were saved in S3 
- used mixed precision training
- 1 epoch was taking 1 hour to run as single gpu max batch size of 256
- trained for 24 epochs (time constraint) and achived an accuarcy of 58%
- IDE used is VS code (windows) connected to ec2 via ssh 


# STEPS to set up project: 

```bash
# 1. activate virtual env

"""
python m venv venv
(conda activate pytorch) if using conda environment
source venv/bin/activate  #(on linux) 
git clone <repo name>

""" 

# 2. install requirements
"""
pip install -r requirements.txt

"""

# 3. Download Imagenet Datasets from kaggle (about 2 hours)
"""
https://www.kaggle.com/c/imagenet-object-localization-challenge/data

"""
"""
pip install kaggle
"""
"""
import kaggle 
"""
"""
dowanload dataset -kaggle competitions download -c im
"""
- configure kaggle your  Kaggle credententials and accept compettion to download the data. 


# push to s3 for Reproducibility(Optional)

"""
aws s3 cp /path/to/your/file.zip s3://your-bucket-name/
"""
# pulling from S3 (ptional)
"""
aws s3 cp  s3://your-bucket-name/  /path/to/your/file.zip
"""


# 4. Unzip the data: (about 45 minutes)
"""
sudo apt install unzip -y
"""
"""
unzip Imagenet.zip

"""

# 5. Modify datasets: will need to modify the validation dataset in correct format first:
- ![alt text](assets/train_data_format.png)             
- ![alt text](assets/val_data_format.png)

"""
python validation_transform.py -d ./Imagenet/ILSVRC/Data/CLS-LOC/val -l ./Imagenet/LOC_val_solution.csv

"""
- ![alt text](assets/data_folder.png)   # after download

# 6. Configure aws to access S3 for saving checkpoints and logs:

"""
sudo apt install update
"""

"""
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"

""" 

"""
sudo apt install unzip -y
"""
"""
unzip awscliv2.zip
"""
"""
sudo ./aws/install
"""

# 7. To begin training:
"""
python train.py
"""
- ![alt text](assets/model_summary.png)

- ![alt text](assets/epoch.png)

- ![alt text](assets/logs_checkpoint.png)

- ![alt text](assets/s3.png)

# 8. Deploy usuing Gradio
- check deploy_gradio_notebook

# 9. Prediction:

- ![alt text](assets/prediction_1.png)       
- ![alt text](assets/prediction_2.png)
- ![alt text](assets/prediction_3.png)      
- ![alt text](assets/prediction_4.png)
- ![alt text](assets/prediction_5.png)    


# 10. Suggestions: 

- train on g5dn 12 x large with multi gpu availablility to reduce training time
- also more memory so can increase bacth size
- mutli gpu is also added in code (in train.py)
- try to run for more epochs to increase accuracy


# 11. Hugging Face Space to test the current version:

https://huggingface.co/spaces/Abhiya/From_Stethoscopes_to_Code

