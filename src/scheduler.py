import torch
import torch.optim as optim

def get_sgd_optimizer(model):
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    return optimizer

def get_one_cycle_LR_scheduler(optimizer, train_loader, epochs):
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(train_loader),
                                        pct_start=0.4, div_factor = 20, three_phase=True, epochs=epochs, verbose=False)
    return scheduler