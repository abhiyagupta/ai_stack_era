import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

##### Get Device Details #####
def get_device() -> tuple:
    """Get Device type

    Returns:
        tuple: Device type
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return (use_cuda, device)

# def move_loss_to_cpu(loss):
#   moved_loss2cpu = [t.cpu().item() for t in loss]
#   return moved_loss2cpu

#####  Get the count of correct predictions
def GetCorrectPredCount(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()


test_incorrect_pred = {"images": [], "ground_truths": [], "predicted_vals": []}


#####  Get the incorrect predictions
def GetInCorrectPreds(pPrediction, pLabels):
    pPrediction = pPrediction.argmax(dim=1)
    indices = pPrediction.ne(pLabels).nonzero().reshape(-1).tolist()
    return indices, pPrediction[indices].tolist(), pLabels[indices].tolist()


def get_incorrect_test_predictions(model, device, test_loader):
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)

            ind, pred, truth = GetInCorrectPreds(output, target)
            test_incorrect_pred["images"] += data[ind]
            test_incorrect_pred["ground_truths"] += truth
            test_incorrect_pred["predicted_vals"] += pred

    return test_incorrect_pred

#####  Display the shape and decription of the train data
def display_train_data(train_data):

  print('[Train]')
  print(' - Numpy Shape:', train_data.cpu().numpy().shape)
  print(' - Tensor Shape:', train_data.size())
  print(' - min:', torch.min(train_data))
  print(' - max:', torch.max(train_data))
  print(' - mean:', torch.mean(train_data))
  print(' - std:', torch.std(train_data))
  print(' - var:', torch.var(train_data))

    
# ### Display the model statistics         
# def display_model_stats(train_loss, train_accuracy, test_loss, test_accuracy):
#   fig, axs = plt.subplots(2,2,figsize=(15,10))
#   axs[0, 0].plot(train_loss)
#   axs[0, 0].set_title("Training Loss")
#   axs[1, 0].plot(train_accuracy)
#   axs[1, 0].set_title("Training Accuracy")
#   axs[0, 1].plot(test_loss)
#   axs[0, 1].set_title("Test Loss")
#   axs[1, 1].plot(test_accuracy)
#   axs[1, 1].set_title("Test Accuracy")