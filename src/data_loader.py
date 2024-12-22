import os
from abc import abstractmethod
from matplotlib import pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2


class dataSet(object):
    mean = None
    std = None
    classes = None

    def __init__(self, batch_size=64, img_data_transforms=None, shuffle=True):
        self.batch_size = batch_size
        self.img_data_transforms = img_data_transforms
        self.shuffle = shuffle
        self.loader_kwargs = {'batch_size': batch_size, 
                              'num_workers': os.cpu_count(), 
                              'pin_memory': True}
        self.train_transforms = self.get_train_transforms()
        self.test_transforms = self.get_test_transforms()
        self.train_loader = self.get_train_loader()
        self.test_loader = self.get_test_loader()
        self.data_iter = iter(self.train_loader)

    def get_train_transforms(self):
        list_train_transforms = [A.Normalize(self.mean, self.std)]
        if self.img_data_transforms is not None:
            list_train_transforms += self.img_data_transforms
        list_train_transforms.append(ToTensorV2())
        return A.Compose(list_train_transforms)

    def get_test_transforms(self):
        list_test_transforms = [A.Normalize(self.mean, self.std), ToTensorV2()]
        return A.Compose(list_test_transforms)

    @abstractmethod
    def get_train_loader(self):
        pass

    @abstractmethod
    def get_test_loader(self):
        pass

    @classmethod
    def de_normalise_image(cls, img):

        for dimg, m, s in zip(img, cls.mean, cls.std):
            dimg.mul_(s).add_(m)
        return img    


    @abstractmethod
    def visualize_transformed_image(self, img):
        pass

    ### Display the CIFAR10 data images
    def show_dataset_images(self, figsize=None, denormalize=True):
        batch_data, batch_label = next(self.data_iter)

        fig = plt.figure(figsize=figsize)
        for i in range(12):
            plt.subplot(3, 4, i + 1)
            plt.tight_layout()
            image = batch_data[i]
            if denormalize:
                image = self.de_normalise_image(image)
            plt.imshow(self.visualize_transformed_image(image), cmap='gray')

            label = batch_label[i].item()
            if self.classes is not None:
                img_title = f'{label}-{self.classes[label]}'
            plt.title(repr(img_title))
            plt.xticks([])
            plt.yticks([])


    ### Display the incorrect predictions of the CIFAR10 data images
    def show_cifar10_incorrect_predictions(self, figsize=None, 
                                           denormalize=False, incorrect_prediction=None):

        for i in range(0, 10):
            print(repr(i) + " - " + repr(self.classes[i]), end=" ")

        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.tight_layout()

            tensor_incorrect_pred = incorrect_prediction["images"][i].cpu().squeeze(0)

            if (denormalize == True):
                img_inv_tensor = self.de_normalise_image(tensor_incorrect_pred)

            plt.imshow(self.visualize_transformed_image(img_inv_tensor), cmap='gray')

            plt.title("Pred " +
                repr(incorrect_prediction["predicted_vals"][i])
                + " vs " + "Truth "
                + repr(incorrect_prediction["ground_truths"][i])
            )
            plt.xticks([])
            plt.yticks([])           