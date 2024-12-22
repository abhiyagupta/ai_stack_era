```
src
| |_data_loader.py                        # Load data
| |_data_preprocessing_albumentation.py   # data tranformation and usuing albumentation Lib
| |_model_architecture.py                 # define model 
| |_model_logs_metrics.py                 # log accuracy train, test 
| |_scheduler.py                          # run train and test
| |_train_test.py                         # plot graphs
| |_utils.py
8_session_cifcar.ipynb                    # run and train model usuing ipynb 
Requirements.txt                          # requirements file

``` 
* Layer Structure - C1 C2 C3 C4 Output + No MaxPooling, all convolution blocks have 3 layers and use 3x3 convolutions, GAP + FC + 1x1 convolutions
* total RF more than 44
* total params less than 200k (190,509))
* used Depthwise Separable Convolution
* used Dilated Convolution
* use GAP 
* Skip connections used
* Batch Normalization and drop out used
* Image Augmentation using Albumentation Transforms - Color Jitter + To Gray + Horizontal flip + Shift Scale Rotate 
* Run 100 epochs
* best epoch 96
* epoch 96 : train accuarcy 86.06 and test accuracy 86.81%


