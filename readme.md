# ASK: 
-mnist dataset - achive accuracy of 99.4% in less then 20K parameters
-less than 20 epochs
-use batch normailization and drop out
-fully connected layer
-github actions



# MODEL ARCHITECTURE: Total Parameters: 15,790
![alt text](image.png) 

# BEST ACCURACY: 98.88%
![alt text](image-1.png)

# INFERENCE RESULTS: 

![alt text](image-2.png)      ![alt text](image-3.png) 

# DETAILS OF PROCESS: 
-created different py files for:
     -model architecture (here tried various optins to get to total params of less then 20,000)
     -used model_6_red_param : where final params are 15,790 
     -creted preprocessing py file. here split dataset into train/test or validation/inference
     -inference set was created without lables and used for final inference or predictions.
     -inference uses randon 10 images from inference dataset and predictiosn are saved in inference_results folder
     - created a github workflow to train and get inference.
     -artifacts saved 
