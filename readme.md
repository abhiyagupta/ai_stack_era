# ASK: 
-mnist dataset - achive accuracy of 99.4% in less then 20K parameters
-less than 20 epochs
-use batch normailization and drop out
-fully connected layer
-github actions



# MODEL ARCHITECTURE: Total Parameters: 15,790
![image](https://github.com/user-attachments/assets/9e8e1da3-f284-4bc4-bcd9-3e637e1992c4)


# BEST ACCURACY: 98.88%
![image](https://github.com/user-attachments/assets/0cfd2abc-71be-4e53-b9ed-70f5c8b07598)


# INFERENCE RESULTS: 
![image](https://github.com/user-attachments/assets/a12f4ed0-d295-4026-b0f7-03d72e2d991f)
![image](https://github.com/user-attachments/assets/fafaccf3-8c65-47a0-a05e-3bff9e6e7099)



# DETAILS OF PROCESS: 
-created different py files for:
     -model architecture (here tried various optins to get to total params of less then 20,000)
     -used model_6_red_param : where final params are 15,790 
     -during training **BEST MODEL** (where acc is max) is saved. this best model is used for predictions. 
     -creted preprocessing py file. here split dataset into train/test or validation/inference
     -inference set was created without lables and used for final inference or predictions.
     -inference uses randon 10 images from inference dataset and predictiosn are saved in inference_results folder
     -created a github workflow to train and get inference.
     -artifacts saved 
