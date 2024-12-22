import time
import math
import torch
import matplotlib.pyplot as plt

torch.manual_seed(1)


class trainModel():
    """ Class to encapsulate model training """

    def __init__(self, model, optimizer,
                 scheduler, train, test,
                 train_loader, test_loader, 
                 lr, epochs, device
                 ):
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train = train
        self.test = test
        self.epochs = epochs
        self.lr = lr
        self.device = device

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.train_loss = []
        self.test_loss = []
        self.train_accuracy = []
        self.test_accuracy = []

        self.schedule = []

        self.start_time = 0
        self.end_time = 0

        self.best_perc = 85.0
        self.best_path = ""

    def epoch_time_period(self):
                
        elapsed_time = self.end_time - self.start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def show_epoch_progress(self, epoch, train_accuracy, train_loss,
                            test_loss, test_accuracy):

        epoch_mins, epoch_secs = self.epoch_time_period()
        learn_rate = self.schedule[epoch]
        print(f'| {epoch+1:5} | {learn_rate:.6f} | {epoch_mins:02}m {epoch_secs:02}s | {train_loss[-1:][0]:.6f}  | {round(train_accuracy[-1:][0], 2):6}%  | {test_loss[-1:][0]:.6f} |{round(test_accuracy[-1:][0], 2):7}% |')
        #print(f'| {epoch+1:5} | {learn_rate:.6f} | {epoch_mins:02}m {epoch_secs:02}s | {train_loss[-1:][0]:.6f} | {str(train_accuracy[-1:][0]):7}% | {test_loss[-1:][0]:.6f} | {str(test_accuracy[-1:][0]):5}% |')


    def save_best_model(self, test_accuracy_percent):

        # = (100.0 * test_accuracy / len(self.test_loader.dataset))

        if test_accuracy_percent >= self.best_perc:
            self.best_perc = test_accuracy_percent
            self.best_path = f'model_weights_{test_accuracy_percent:.2f}.pth'
            torch.save(self.model.state_dict(), self.best_path)

    def run_training_model(self):
        
        print(
            f'| Epoch | {"LR":8} | {"Time":7} | {"TrainLoss":7} | {"TrainAcc":7} | {"TestLoss":7} | {"TestAcc":7} |')

        for epoch in range(self.epochs):
            self.schedule.append(self.optimizer.param_groups[0]['lr'])

            self.start_time = time.time()

            train_loss, train_accuracy = self.train(model=self.model,
                                                    device=self.device,
                                                    train_loader=self.train_loader, 
                                                    optimizer=self.optimizer, 
                                                    scheduler=self.scheduler)
            
            test_loss, test_accuracy = self.test(model=self.model,
                                                 device=self.device,
                                                 test_loader=self.test_loader)

            
            self.end_time = time.time()
            self.save_best_model(test_accuracy[-1:][0])

            self.show_epoch_progress(epoch, train_accuracy, train_loss, 
                                      test_loss, test_accuracy)
        
        self.train_loss =  train_loss
        self.test_loss = test_loss
        self.train_accuracy = test_accuracy
        self.test_accuracy = test_accuracy


    def show_best_model(self):

        self.model.load_state_dict(torch.load(self.best_path))
        self.model.eval()

        test_loss, test_accuracy = self.test(self.model, self.device, self.test_loader)

        print(f'Test Accuracy: {test_accuracy[-1:][0]:.2f}% | Test Loss: {test_loss[-1:][0]:.6f}')


    ### Display the model statistics         
    def display_model_stats(self):
        fig, axs = plt.subplots(2,2,figsize=(15,10))
        train_loss = [t.cpu().item() for t in self.train_loss]
        #tr_loss = move_to_cpu(self.train_loss)
        axs[0, 0].plot(train_loss)
        axs[0, 0].set_title("Training Loss")
        axs[1, 0].plot(self.train_accuracy)
        axs[1, 0].set_title("Training Accuracy")
        axs[0, 1].plot(self.test_loss)
        axs[0, 1].set_title("Test Loss")
        axs[1, 1].plot(self.test_accuracy)
        axs[1, 1].set_title("Test Accuracy")