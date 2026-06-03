import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time, sys
from util.time import *
from util.env import *
from test import *
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR




def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')

    return loss



def train(model = None, save_path = '', config={},  train_dataloader=None, val_dataloader=None):
    
    seed = config['seed']

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=config['decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max = config['epoch'], eta_min = 1e-5)
    now = time.time()
    
    train_loss_list = []
    val_loss_list = []
    lr_list = []
    
    device = get_device()

    acu_loss = 0
    min_loss = 1e+20

    i = 0
    epoch = config['epoch']
    early_stop_win = 10000

    model.train()

    stop_improve_count = 0

    dataloader = train_dataloader
    scaler = GradScaler()  # use this scaler to do AMP (Automatic Mixed Precision) - speeds up training 
    for i_epoch in range(epoch):
        acu_loss = 0
        model.train()

        batch_losses = []
        lr_list.append(optimizer.param_groups[0]['lr'])
        for x, labels, _, edge_index in dataloader:
            _start = time.time()
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            edge_index = edge_index.to(device, non_blocking=True)
            _end = time.time()
            #print(f"Time for batch_loading: {_end-_start}")
            optimizer.zero_grad()

            _start = time.time()
            with autocast():
                out, learned_graph = model(x, edge_index)
                _end = time.time()
                #print(f"Time for forward pass: {_end-_start} ")
                loss = loss_func(out, labels)
            
            _start = time.time()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            _end = time.time()
            #print(f"Time for backward loss and optimizer.step() {_end - _start}")
            
            batch_losses.append(loss.item())

            
            acu_loss += loss.item()
                
            i += 1

        avg_train_loss = sum(batch_losses) / len(batch_losses)
        train_loss_list.append(avg_train_loss)

        # each epoch
        print('epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})'.format(
                        i_epoch + 1, epoch, 
                        avg_train_loss, acu_loss), flush=True
            )

        # use val dataset to judge
        if val_dataloader is not None:

            model.eval()
            val_loss_list_batch = []

            with torch.no_grad():
                for x_val, y_val, labels_val, edge_index_val in val_dataloader:
                    _start = time.time()
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    edge_index_val = edge_index_val.to(device)
                    _end = time.time()
                    #print(f"Time for val batch_loading: {_end-_start}")
                    
                    with autocast():
                        _start = time.time()
                        val_pred, _ = model(x_val, edge_index_val)
                        v_loss = loss_func(val_pred, y_val)
                    val_loss_list_batch.append(v_loss.item())
                    _end = time.time()
                    #print(f"Time for val forward pass {_end-_start}")

                scheduler.step() # Apply schedular step for learning rate

            val_loss = sum(val_loss_list_batch) / len(val_loss_list_batch)
            val_loss_list.append(val_loss)
            
            print(f"Validation Loss: {val_loss:.8f}")


            if val_loss < min_loss:
                _start = time.time()
                torch.save(model.state_dict(), save_path)
                _end = time.time()
                #print(f"Time for saving model: {_end-_start}")
                min_loss = val_loss 
                stop_improve_count = 0
            else:
                stop_improve_count += 1


            if stop_improve_count >= early_stop_win:
                break

        else:
            val_loss_list.append(None)  # if no val, just append None
            if acu_loss < min_loss :
                torch.save(model.state_dict(), save_path)
                min_loss = acu_loss
        

    torch.save(learned_graph.detach().cpu(), f"_analysis/graph/{config['comment']}.pt")

    return train_loss_list, val_loss_list, lr_list
