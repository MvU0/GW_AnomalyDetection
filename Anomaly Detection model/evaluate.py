from util.data import *
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score


def get_full_err_scores(test_result, val_result, run_mean_window, sample_len, slide_stride):
    np_test_result = np.array(test_result)
    np_val_result = np.array(val_result)


    all_scores =  None
    all_normals = None
    feature_num = np_test_result.shape[-1]

    labels = np_test_result[2, :, 0].tolist()


    # For validation set, sampling may be done in intervals to match training (example sample_win = 5 for training/val and always 1 for test)
    # So need to adjust sample_len for validation set
    sample_len_val = np.floor(sample_len/slide_stride)

    for i in range(feature_num):
        test_re_list = np_test_result[:2,:,i] # Here take only first two elements in zero-th dimension
        val_re_list = np_val_result[:2,:,i]  # Because index 0 is predicted_value, index 1 is ground_truth_value, and index 2 is label

        scores = get_err_scores(test_re_list, run_mean_window, sample_len)
        normal_dist = get_err_scores(val_re_list, run_mean_window, sample_len_val)

        if all_scores is None:
            all_scores = scores
            all_normals = normal_dist
        else:
            all_scores = np.vstack((
                all_scores,
                scores
            ))
            all_normals = np.vstack((
                all_normals,
                normal_dist
            ))

    return all_scores, all_normals


def get_final_err_scores(test_result, val_result):
    full_scores, all_normals = get_full_err_scores(test_result, val_result, return_normal_scores=True)

    all_scores = np.max(full_scores, axis=0)

    return all_scores



def get_err_scores(test_res, run_mean_window, sample_len):
    "Compute normalized and smoothed error scores"

    test_predict, test_gt = test_res

    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)

    test_delta = np.abs(np.subtract(
                        np.array(test_predict).astype(np.float64), 
                        np.array(test_gt).astype(np.float64)
                    ))
    epsilon=1e-2

    err_scores = (test_delta - n_err_mid) / ( np.abs(n_err_iqr) +epsilon)


    total_len = len(err_scores)
    if total_len % sample_len != 0:
        raise ValueError(
            f"Total length ({total_len}) is not divisible"
            f"by sample_len ({sample_len})"
        )
    num_samples = total_len // sample_len

    # Reshape data to make be sure average anomaly score doesn't get taken across samples
    err_scores = err_scores.reshape(num_samples, sample_len)   

    #Apply moving average independently per sample
    smoothed_err_scores = moving_average_per_sample(err_scores, run_mean_window)

    # The moving_average_per_sample function gives all zeros for the first run_mean_window timesteps, so need to get rid of those
    smoothed_err_scores = smoothed_err_scores[:, run_mean_window:].reshape(-1)
   
    return smoothed_err_scores

def moving_average_per_sample(arr, window):
    "Apply moving average independently to each sample"

    kernel = np.ones(window + 1) / (window + 1)
    # Output array initialized to zeros
    out = np.zeros_like(arr)
    for s in range(arr.shape[0]):

        # Moving average using convolution
        conv = np.convolve(arr[s], kernel, mode='valid')

        out[s, window:] = conv

    return out


def process_labels(labels, sample_len, run_mean_window):
    """
    Remove test labels at the beginning of each sample to account for moving average in anomaly score.
    """
    labels = np.array(labels)

    total_len = len(labels)

    if total_len % sample_len != 0:
        raise ValueError(
            f"Total label length ({total_len}) is not divisible "
            f"by sample_len ({sample_len})"
        )

    num_samples = total_len // sample_len
    # Reshape into samples
    labels = labels.reshape(num_samples, sample_len)

    # Remove first `run_mean_window` labels per sample and reshape back into original form
    labels = labels[:, run_mean_window:].reshape(-1)

    return labels.tolist()


def get_loss(predict, gt):
    return eval_mseloss(predict, gt)


def get_f1_scores(total_err_scores, gt_labels, topk=1):
    
    # remove the highest and lowest score at each timestep
    total_features = total_err_scores.shape[0]

    # topk_indices = np.argpartition(total_err_scores, range(total_features-1-topk, total_features-1), axis=0)[-topk-1:-1]
    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]
    
    topk_indices = np.transpose(topk_indices)

    total_topk_err_scores = []
    topk_err_score_map=[]
    # topk_anomaly_sensors = []

    for i, indexs in enumerate(topk_indices):
       
        sum_score = sum( score for k, score in enumerate(sorted([total_err_scores[index, i] for j, index in enumerate(indexs)])) )

        total_topk_err_scores.append(sum_score)

    final_topk_fmeas = eval_scores(total_topk_err_scores, gt_labels, 400)

    return final_topk_fmeas


def get_val_performance_data(total_err_scores, normal_scores, gt_labels, topk=1):
    total_features = total_err_scores.shape[0]
    
    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]

    total_topk_err_scores = []
    topk_err_score_map=[]

    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)

    threshold = np.max(normal_scores)

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > threshold] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)

    f1 = f1_score(gt_labels, pred_labels)


    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    return f1, pre, rec, auc_score, threshold, pred_labels, gt_labels, total_topk_err_scores, topk_indices


def get_best_performance_data(total_err_scores, gt_labels, topk=1):

    total_features = total_err_scores.shape[0]

    # topk_indices = np.argpartition(total_err_scores, range(total_features-1-topk, total_features-1), axis=0)[-topk-1:-1]
    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]

    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)

    final_topk_fmeas ,thresolds = eval_scores(total_topk_err_scores, gt_labels, 400, return_thresold=True)

    th_i = final_topk_fmeas.index(max(final_topk_fmeas))
    thresold = thresolds[th_i]

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)

    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    return max(final_topk_fmeas), pre, rec, auc_score, thresold

