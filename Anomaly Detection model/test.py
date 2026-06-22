import torch
import torch.nn as nn
import time, sys
from util.time import *
from util.env import *
from torch.cuda.amp import autocast

from util.data import *
from util.preprocess import *



def test(model, dataloader):
    loss_func = nn.MSELoss(reduction='mean')
    device = get_device()

    test_loss_list = []

    preds = []
    gts = []
    labs = []

    model.eval()

    i = 0

    for x, y, labels, edge_index in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        edge_index = edge_index.to(device, non_blocking=True)

        with torch.no_grad():
            with autocast():
                predicted, _ = model(x, edge_index)
                loss = loss_func(predicted, y)

            labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])

            preds.append(predicted.detach().cpu())
            gts.append(y.detach().cpu())
            labs.append(labels.detach().cpu())

        test_loss_list.append(loss.item())
        i += 1

    t_test_predicted_list = torch.cat(preds, dim=0)
    t_test_ground_list = torch.cat(gts, dim=0)
    t_test_labels_list = torch.cat(labs, dim=0)

    avg_loss = sum(test_loss_list) / len(test_loss_list)

    return avg_loss, [t_test_predicted_list, t_test_ground_list, t_test_labels_list], [t_test_predicted_list, t_test_ground_list]