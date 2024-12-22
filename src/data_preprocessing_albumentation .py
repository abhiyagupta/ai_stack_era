import numpy as np
import torch
import torchvision
import albumentations as A

from .data_loader import dataSet

torch.manual_seed(11)

#### Class to transform CIFAR10 Data Set with Albumentations
class albumentationTransforms(torchvision.datasets.CIFAR10):
    def __init__(self, root, data_alb_transform=None, **kwargs):
        super(albumentationTransforms, self).__init__(root, **kwargs)
        self.data_alb_transform = data_alb_transform

    def __getitem__(self, index):
        trans_image, trans_label = super(albumentationTransforms, self).__getitem__(index)
        if self.data_alb_transform is not None:
            trans_image = self.data_alb_transform(image=np.array(trans_image))['image']
        return trans_image, trans_label
    

#### CIFAR10 Data Set Class
class cifar10Set(dataSet):
    mean = (0.49139968, 0.48215827, 0.44653124)
    std = (0.24703233, 0.24348505, 0.26158768)
    # denorm_mean = (-0.491, -0.482, -0.446)
    # denorm_std = (1/0.247, 1/0.243, 1/0.261)
    classes = None

    def get_train_transforms(self):
        if self.img_data_transforms is None:
            self.img_data_transforms = [

                A.ColorJitter(brightness=0, contrast=0.1, 
                              saturation=0.2, hue=0.1, p=0.4),
                A.ToGray(p=0.1),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, 
                                   rotate_limit=15),
                # Since normalisation was the first step, mean is already 0, 
                # so cutout fill = 0
                A.CoarseDropout(max_holes=1, 
                                max_height=16, 
                                max_width=16, p=0.2, 
                                fill_value=0)
            ]
        return super(cifar10Set, self).get_train_transforms()

    ### Load the train data as to be used by the model
    def get_train_loader(self):
        super(cifar10Set, self).get_train_loader()

        train_data = albumentationTransforms('./data', 
                                             train=True, 
                                             download=True, 
                                             data_alb_transform=self.train_transforms)
        # train dataloader    
        if self.classes is None:
            self.classes = {i: c for i, c in enumerate(train_data.classes)}
        self.train_loader = torch.utils.data.DataLoader(train_data, 
                                                        shuffle=self.shuffle, 
                                                        **self.loader_kwargs)
        return self.train_loader
    
    ### Load the train and test data as to be used by the model 
    def get_test_loader(self):
        super(cifar10Set, self).get_test_loader()

        test_data = albumentationTransforms('./data', 
                                            train=False, 
                                            download=True,
                                            data_alb_transform=self.test_transforms)

        # test dataloader    
        self.test_loader = torch.utils.data.DataLoader(test_data, 
                                                       shuffle=False, 
                                                       **self.loader_kwargs)
        return self.test_loader

    def visualize_transformed_image(self, img):
        return img.permute(1, 2, 0)