# util functions about data

from scipy.stats import rankdata, iqr, trim_mean
from sklearn.metrics import f1_score, mean_squared_error
import numpy as np
from numpy import percentile
import pandas as pd
import h5py


def get_subsys_masks(channel_names):
    # Create boolean masks for each subsystem
    subsys_counts = {}
    subsys_masks = {}
    n = len(channel_names)
    for i, name in enumerate(channel_names):
        key = name[3:6]

        subsys_counts[key] = subsys_counts.get(key, 0) + 1

        if key not in subsys_masks:
            subsys_masks[key] = np.zeros(n, dtype=bool)
        subsys_masks[key][i] = True

    
    channel_to_subsystem_full = {}
    for subsystem, mask in subsys_masks.items():
        indices = np.where(mask)[0]
        for idx in indices:
            channel_to_subsystem_full[channel_names[idx]] = subsystem
    
    return subsys_masks, channel_to_subsystem_full

def get_attack_interval(attack): 
    heads = []
    tails = []
    for i in range(len(attack)):
        if attack[i] == 1:
            if attack[i-1] == 0:
                heads.append(i)
            
            if i < len(attack)-1 and attack[i+1] == 0:
                tails.append(i)
            elif i == len(attack)-1:
                tails.append(i)
    res = []
    for i in range(len(heads)):
        res.append((heads[i], tails[i]))
    # print(heads, tails)
    return res

# calculate F1 scores
def eval_scores(scores, true_scores, th_steps, return_thresold=False):
    padding_list = [0]*(len(true_scores) - len(scores))
    # print(padding_list)

    if len(padding_list) > 0:
        scores = padding_list + scores

    scores_sorted = rankdata(scores, method='ordinal')
    th_steps = th_steps
    # th_steps = 500
    th_vals = np.array(range(th_steps)) * 1.0 / th_steps
    fmeas = [None] * th_steps
    thresholds = [None] * th_steps
    for i in range(th_steps):
        cur_pred = scores_sorted > th_vals[i] * len(scores)

        fmeas[i] = f1_score(true_scores, cur_pred)

        score_index = scores_sorted.tolist().index(int(th_vals[i] * len(scores)+1))
        thresholds[i] = scores[score_index]

    if return_thresold:
        return fmeas, thresholds
    return fmeas

def eval_mseloss(predicted, ground_truth):

    ground_truth_list = np.array(ground_truth)
    predicted_list = np.array(predicted)

    
    # mask = (ground_truth_list == 0) | (predicted_list == 0)

    # ground_truth_list = ground_truth_list[~mask]
    # predicted_list = predicted_list[~mask]

    # neg_mask = predicted_list < 0
    # predicted_list[neg_mask] = 0

    # err = np.abs(predicted_list / ground_truth_list - 1)
    # acc = (1 - np.mean(err))

    # return loss
    loss = mean_squared_error(predicted_list, ground_truth_list)

    return loss

def get_err_median_and_iqr(predicted, groundtruth):

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = np.median(np_arr)
    err_iqr = iqr(np_arr)

    return err_median, err_iqr

def get_err_median_and_quantile(predicted, groundtruth, percentage):

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = np.median(np_arr)
    # err_iqr = iqr(np_arr)
    err_delta = percentile(np_arr, int(percentage*100)) - percentile(np_arr, int((1-percentage)*100))

    return err_median, err_delta

def get_err_mean_and_quantile(predicted, groundtruth, percentage):

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = trim_mean(np_arr, percentage)
    # err_iqr = iqr(np_arr)
    err_delta = percentile(np_arr, int(percentage*100)) - percentile(np_arr, int((1-percentage)*100))

    return err_median, err_delta

def get_err_mean_and_std(predicted, groundtruth):

    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_mean = np.mean(np_arr)
    err_std = np.std(np_arr)

    return err_mean, err_std


def get_f1_score(scores, gt, contamination):

    padding_list = [0]*(len(gt) - len(scores))
    # print(padding_list)

    threshold = percentile(scores, 100 * (1 - contamination))

    if len(padding_list) > 0:
        scores = padding_list + scores

    pred_labels = (scores > threshold).astype('int').ravel()

    return f1_score(gt, pred_labels)




def save_scores_to_hdf5(scores, base_path, segment_starts=None):
    """
    Save scores (numpy array) to HDF5, optionally split into segments.
    """

    scores = np.asarray(scores, dtype=np.float32)  # ensure numeric

    hdf5_path = f"{base_path}.h5"

    # Prepare segment boundaries
    if segment_starts is not None:
        segment_starts = sorted(segment_starts)
        segment_starts = list(segment_starts) + [scores.shape[1]]  # time axis
    else:
        segment_starts = [0, scores.shape[1]]

    with h5py.File(hdf5_path, "w") as f:

        for i in range(len(segment_starts) - 1):
            start, end = segment_starts[i], segment_starts[i+1]

            segment = scores[:, start:end]  # slice along time axis

            grp = f.create_group(f"Segment_{i}")
            grp.create_dataset(
                name="scores",
                data=segment,
                compression="gzip"
            )
    
    print(f"HDF5 saved to {hdf5_path}")
