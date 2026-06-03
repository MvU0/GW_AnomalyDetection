import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time, sys
from util.time import *
from util.env import *

import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F
from torch.cuda.amp import autocast

from util.data import *
from util.preprocess import *



def test(model, dataloader):
    # test
    loss_func = nn.MSELoss(reduction='mean')
    device = get_device()

    test_loss_list = []
    now = time.time()

    test_predicted_list = []
    test_ground_list = []
    test_labels_list = []

    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []

    test_len = len(dataloader)

    model.eval()

    i = 0
    acu_loss = 0
    for x, y, labels, edge_index in dataloader:
        _start = time.time()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        edge_index = edge_index.to(device, non_blocking=True)
        _end = time.time()
        #print(f"Time for test batch loading: {_end-_start}")

        _start = time.time()
        with torch.no_grad():
            with autocast():
                predicted, _ = model(x, edge_index)
             
                loss = loss_func(predicted, y)
            
            _end = time.time()
            #print(f"Forward test pass: {_end-_start}")
            labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])

            _start = time.time()
            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
                t_test_ground_list = y
                t_test_labels_list = labels
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)
        
        test_loss_list.append(loss.item())
        acu_loss += loss.item()
        
        i += 1
        _end = time.time()
        #print(f'Putting data into list {_end-_start}')
        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))

    
    test_predicted_list = t_test_predicted_list.tolist()        
    test_ground_list = t_test_ground_list.tolist()        
    test_labels_list = t_test_labels_list.tolist()      
    
    avg_loss = sum(test_loss_list)/len(test_loss_list)

    return avg_loss, [test_predicted_list, test_ground_list, test_labels_list], [t_test_predicted_list, t_test_ground_list]




