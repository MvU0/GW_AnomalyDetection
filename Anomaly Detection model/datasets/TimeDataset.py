import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

import sys


class TimeDataset(Dataset):
    def __init__(self, raw_data, edge_index, mode='train', config=None, segment_length=2048):
        data = np.array(raw_data[:-1], dtype=np.float32)
        self.raw_data = torch.from_numpy(data)  # features
        labels = np.array(raw_data[-1], dtype=np.float32)
        self.labels = torch.from_numpy(labels)    # labels
        self.edge_index = edge_index.long()
        self.mode = mode
        self.config = config
        self.segment_length = segment_length

        self.slide_win, self.slide_stride = [self.config[k] for k in ['slide_win', 'slide_stride']]
        self.node_num, self.total_time_len = self.raw_data.shape

        # Precompute valid indices (optional, especially if segments exist)
        self.valid_indices = self._compute_valid_indices()

    def _compute_valid_indices(self):
        indices = []
        rng = range(self.slide_win, self.total_time_len, self.slide_stride) \
            if self.mode == 'train' else range(self.slide_win, self.total_time_len)
        
        for i in rng:
            start_ft_seg = (i - self.slide_win) // self.segment_length   # Here check for each index i if the target and the window used to predict it are in the same segment
            target_seg = i // self.segment_length

            if start_ft_seg == target_seg:
                indices.append(i)

        return indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]  # actual time index for this sample

        # compute window on-the-fly
        ft = self.raw_data[:, i - self.slide_win:i]    # shape: [nodes, slide_win]
        tar = self.raw_data[:, i]                      # shape: [nodes]
        label = self.labels[i]                          # scalar

        return ft, tar, label, self.edge_index
