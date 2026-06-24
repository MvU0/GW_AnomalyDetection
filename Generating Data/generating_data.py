import numpy as np
import pandas as pd
import h5py
import scipy.signal as signal
from gwpy.timeseries import TimeSeries as GWpyTimeSeries
from gwpy.frequencyseries import FrequencySeries as GWpyFrequencySeries
import json
import os
import sys
import time

"""
This code is made to instantly go from .csv files that point to .h5 files which contain actual timeseries data to ready to use parquet files.
The different channels will be downsampled such that all timeseries have a sampling rate of 256 Hz
Parquet files are grouped by class, and can be whitened, normalized/standardized. 
The data is made into one long 2D dataset of shape (N_samples*timesteps_per_sample, channels).
Each sample is cut off at eight seconds (minimum size of the samples), such that each sample has the same size.

"""


# ================== GLOBAL VARIABLES ================== #
ALL_CLASSES = ("Blip", "Blip_Low_Frequency", "Clean", "Extremely_Loud", "Fast_Scattering", "Koi_Fish", "Low_Frequency_Burst", "Low_Frequency_Lines", "No_Glitch", "Scattered_Light", "Tomte", "Whistle")
ALL_DATASETS = ("O3a", "O3b")

def data_from_csv(class_csvs = None, whiten=False, ASD_dict=None):
    """Loads time-series data from CSV references to HDF5 files, downsamples all channels to 256 Hz, 
    optionally whitens the data using a supplied ASD, and returns arrays grouped by class together with channel names."""


    # Here the timeseries data is stored in the form of .h5 files for LIGO-L1 data with GPS-times in the name
    dir_to_data = "/data/gravwav/lopezm/Projects/AnomalyDetection/data_products/timeseries/full/" 

    # This folder contains csv files per glitch class pointing to the .h5 files 
    dir_to_csvs = "/data/gravwav/lopezm/Projects/AnomalyDetection/data_products/timeseries/matches/"
    full_csv_paths = [dir_to_csvs + file for file in class_csvs]

    class_data = {}  # <-- store data per class

    channel_names = None
    n_channels = None
    for class_idx, class_csv_path in enumerate(full_csv_paths):
        matches = pd.read_csv(class_csv_path)
        files_start_arr = matches["file_start"].to_numpy()
        seg_start_arr = matches["seg_start"].to_numpy()
        data_files = matches["file"].str.replace(".gwf", "_DLIST.h5").to_numpy() # The csv files point to .gwf files but on NIKHEF they are stored in .h5 files
        if len(data_files) == 0:
            print(f"Empty CSV: {class_csv_path}", flush=True)
            return
        
        # Each csv file points to a bunch of different .h5 files all representing the same class of glitch (or clean data) at different times
        # Following block checks if each .h5 file actually exists and if it doesn't goes to the next file

        temp_file = None
        valid_file_path = None

        for f in data_files:
            candidate_path = os.path.join(dir_to_data, f)
            if not os.path.exists(candidate_path):
                continue
            try:
                temp_file = h5py.File(candidate_path, 'r')
                valid_file_path = candidate_path
                break
            except (OSError, IOError):
                continue
        if temp_file is None:
            raise FileNotFoundError("No valid HDF5 files found in provided file list")
        
        print(f"Using file for metadata: {valid_file_path}", flush=True)

        if channel_names is None: # Only need to get the channel names and sampling rates once
            channel_names_ts = list(temp_file["timeseries"].keys())
            channel_names_md = [ch.decode() if isinstance(ch, bytes) else ch 
                                for ch in temp_file["metadata"]["channels"]]
            sampling_rate = list(temp_file["metadata"]["sample_rate"])
            sampling_min = min(sampling_rate)
            sampling_ratio = (sampling_rate // sampling_min).astype(int)
            sr_dict = {ch: sampling_ratio[i] for i, ch in enumerate(channel_names_md)}
            
            # Detect channels which are fully zero/constant -- the code assumes this will be constant across all classes
            channel_names = []
            for ch in channel_names_ts:
                if np.std(temp_file["timeseries"][ch][:]) != 0:
                    channel_names.append(ch)
            n_channels = len(channel_names)
            
            temp_file.close()


        

        # Prepare to store each segment as array
        segment_arrays = np.zeros((len(data_files), 2048, n_channels), dtype=np.float32)
        files_to_close_to_boundary = 0

        for file_idx in range(len(data_files)):

            file_path = dir_to_data + data_files[file_idx]
            try:
                f_h5 = h5py.File(file_path, "r")
            except (FileNotFoundError, OSError):
                print(f"Skipping missing or unreadable file: {file_path}", flush=True)
                continue
            
            file_start = files_start_arr[file_idx] # The data files contain timesteps before and after a stretch of data was flagged as a certain class
            seg_start = seg_start_arr[file_idx]    # File start and segment start are given in GPS times. File start is the GPS time where the data file starts
                                                  # while seg_start is the GPS time where the flagged period starts with a ceratin class 


            relative_start = seg_start - file_start
            if (relative_start - 0.5) < 0 or (relative_start + 8.5) >= 64:      # Check if segment of interest is too close to edge of full file. Need half a second boundary on each side to prevent edge effects with downsampling
                files_to_close_to_boundary += 1
                continue

            start_index = int(np.ceil((seg_start - file_start) * sampling_min)) # Need to calculate the start_index. It will be the same across all channels after downsampling
                                                                                # Is constant for all channels with the minimum sampling rate (256) so can calculate at the start once
            

            # Collect one segment array (minimum_seg_length = 2048 x n_channels) 
            segment = np.zeros((2048, n_channels), dtype=np.float32)    # HERE SOMETHING SHOULD CHANGE IF I WANT VARIABLE SEGMENT LENGTHS
            
            timeseries = f_h5["timeseries"]  # loading this once first, avoids having to go through the look-up everytime in ch_data = ..
            channel_dsets = [timeseries[ch] for ch in channel_names]

            for j, ch_data in enumerate(channel_dsets):
                
                ch = channel_names[j]
                ratio = sr_dict[ch]  # Do it like this because maybe looking up the value the whole time instead of saving it takes a surprisingly long time
                
                if ratio > 1:
                    # Take 9 seconds in total - start -0.5 seconds before start and go until 0.5 seconds after segment to get rid of border effects in decimating
                    specific_start_index = int(np.ceil(((seg_start-0.5) - file_start)*sampling_min*ratio)) 
                    specific_end_index = int(int(np.ceil(((seg_start+0.5) - file_start)*sampling_min*ratio)) + 2048*ratio)

                    seg_of_interest = ch_data[specific_start_index:specific_end_index]
                    long_seg = signal.decimate(seg_of_interest, ratio, zero_phase=True)
                    seg = long_seg[128:128+2048]  # 128 is 0.5 seconds at sampling rate of 256 so need to crop that still
                else:
                    seg = ch_data[start_index:start_index + 2048]
                

                if whiten == True:
                    asd = ASD_dict[ch].value

                    data_fft = np.fft.rfft(seg)
                    white_fft = data_fft / asd
                    seg = np.fft.irfft(white_fft)
              
                    segment[:,j] = seg 
                else:
                    segment[:,j] = seg

            segment_arrays[file_idx] = segment
            f_h5.close()
        
            if file_idx % 10 == 0:
                print(f"{file_idx}/{len(data_files)} files processed", flush=True)


        mask = ~(np.all(segment_arrays == 0, axis=(1,2)))
        segment_arrays = segment_arrays[mask]
        class_name = os.path.basename(class_csv_path).replace(".csv", "")
        class_data[class_name] = segment_arrays    
    print('Files disregarded because they were too close to the edges of the full segment:', files_to_close_to_boundary, flush=True)
    return class_data, channel_names


def preprocess_data(data, classes, percentage_preprocess, mode):
    "Applies dataset-wide standardization or normalization using statistics computed from a configurable subset of the data."
    "Possible modes are: 'standardize' or 'normalize'."
    "Data comes in the shape of a dictionary"

    if isinstance(percentage_preprocess, (int,float)):
        perc_prep = [percentage_preprocess for cls in classes]   # Turn percentage_preprocess into a list so it can be looped over in case it was a single float at the start.
    else:
        perc_prep = percentage_preprocess

    train_dataset = []
    for idx, class_name in enumerate(classes):
        temp_data = data[class_name]
        if isinstance(temp_data, dict):
            full = np.concatenate([temp_data['train'], temp_data['test']], axis=0)
        else:
            full = temp_data
        train_dataset.append(full[:int(full.shape[0]*perc_prep[idx]),:,:]) # CHECK! NEED TO MAKE THE THING AN INTEGER OTHERWISE INDEXING GOES BAD
    train_dataset = np.concatenate(train_dataset, axis=0)

    prep_dict = {}
    original_classes = list(data.keys())
    if mode == 'standardize':
        mean = np.mean(train_dataset, axis=(0,1))
        std = np.std(train_dataset, axis=(0,1))

        for class_name in original_classes:
            class_data = data[class_name]
            if isinstance(class_data, dict):
                train = (class_data['train'] - mean) / std
                test = (class_data['test'] - mean) / std
                
                prep_dict[class_name] = {}  # initialize the dict first
                prep_dict[class_name]['train'] = train
                prep_dict[class_name]['test'] = test
            else:
                full = (class_data - mean) / std
                prep_dict[class_name] = full
    
    if mode == 'normalize':
        min = np.min(train_dataset, axis=(0,1))
        max = np.max(train_dataset, axis=(0,1))

        for class_name in original_classes:
            class_data = data[class_name]
            if isinstance(class_data, dict):
                train = (class_data['train'] - min) / (max - min)
                test = (class_data['test'] - min) / (max - min)

                prep_dict[class_name] = {}  # initialize the dict first
                prep_dict[class_name]['train'] = train
                prep_dict[class_name]['test'] = test
            else:
                full = (class_data - min) / (max - min)
                prep_dict[class_name] = full

    return prep_dict
    

def ASD_calc(data, percentage_whitening, channel_names):
    """Builds train/test splits for whitening, estimates an average ASD per channel from the training data, 
    and returns both the ASD dictionary and whitened datasets."""
    "data comes in the form of a dictionary"
    "percentage_whitening must either be the same length as data.keys() or one value --> Check this at start of main function  CHECK!"

    # ==== Constants ==== #
    seg_len = 1
    overlap = seg_len / 2
    fs = 256
    method = "median" # method='median' means taking the median of the segments in PSD calculation - other option is 'welch' which takes mean of the segments


    class_names = list(data.keys()) # The keys in the dictionary are the class names

    if isinstance(percentage_whitening, float):
        ratio_list = [percentage_whitening for name in class_names]
    else:
        ratio_list = percentage_whitening
        
    psd_dict = {}
    unwhitened_data_dict = {}
    mean_asd_dict = {}
    

    # ======== SPLIT TRAIN / TEST ========
    for class_name_idx, ratio in enumerate(ratio_list):    # Dictionary keys keeps original ordering

        class_name = class_names[class_name_idx]
        class_data = data[class_name]

        split_idx = int(ratio * len(class_data))

        train = class_data[:split_idx,:,:]
        test = class_data[split_idx:,:,:]
        
        unwhitened_data_dict[f"{class_name}_train"] = train
        unwhitened_data_dict[f"{class_name}_test"] = test

    # ======== VECTORISED PSD ========
    # Here calculate PSD in a vectorised way - supposedly a lot faster than creating a bunch of GWPy TimeSeries and doing PSD calculation one at a time -- can do a bunch at the same time
    example_class = class_names[0]
    train_example = unwhitened_data_dict[f"{example_class}_train"]

    n_samples, timesteps, n_channels = train_example.shape
    for ch_idx, channel in enumerate(channel_names):

        channel_stack = []

        for class_name in class_names:

            train = unwhitened_data_dict[f"{class_name}_train"]
            channel_stack.append(train[:,:,ch_idx])

        channel_data = np.concatenate(channel_stack, axis=0) # Turn the data per sample across channels into a numpy array

        freqs, psd = signal.welch(channel_data, fs=fs, window="hann", 
                                  nperseg=int(fs*seg_len), noverlap=int(fs*overlap),
                                  detrend="constant", scaling="density",
                                  average=method, axis=1)

        mean_psd = np.mean(psd, axis=0)
        freq_fft = np.fft.rfftfreq(2048, 1/fs)  # This is the frequency grid of the final PSD you want.      
        psd_interp = np.interp(freq_fft, freqs, mean_psd)    
        

        mean_asd_dict[channel] = GWpyFrequencySeries(
            data=np.sqrt(psd_interp),
            frequencies=freq_fft
        )
        
    # ======== VECTORIZED WHITENING ========
    
    whitened_data_dict = {}
    
    for class_name in class_names:

        whitened_data_dict[class_name] = {}

        for split in ["train", "test"]:
            data_array = unwhitened_data_dict[f"{class_name}_{split}"]

            n_samples, timesteps, n_channels = data_array.shape
            whitened = np.zeros_like(data_array, dtype=np.float32)

            for ch_idx, ch_name in enumerate(channel_names):
                
                asd = mean_asd_dict[ch_name].value

                channel_data = data_array[:, :, ch_idx]

                data_fft = np.fft.rfft(channel_data, axis=1)
                white_fft = data_fft / asd
                white = np.fft.irfft(white_fft, axis=1)

                whitened[:, :, ch_idx] = white
            
            whitened_data_dict[class_name][split] = whitened
    return mean_asd_dict, whitened_data_dict


def save_dataset_info(save_dir, classes, datasets, whiten, class_for_whitening, datasets_for_whitening,  percentage_whitening,
                      preprocess, preprocess_mode, class_for_preprocessing, datasets_for_preprocessing, percentage_preprocess):
    """Stores the configuration used to generate the dataset in a JSON metadata file."""
    info = {}

    info["classes_in_dataset"] = list(classes)
    info["datasets"] = [datasets] if isinstance(datasets, str) else list(datasets)

    # Whitening section
    whitening_info = {}
    whitening_info["enabled"] = whiten
    whitening_info["datasets"] = [datasets_for_whitening] if isinstance(datasets_for_whitening, str) else list(datasets_for_whitening)

    if whiten:
        if isinstance(class_for_whitening, str):
            whitening_info["classes"] = {
                class_for_whitening: percentage_whitening
            }
        else:
            whitening_info["classes"] = {
                cls: percentage_whitening for cls in class_for_whitening
            }

    info["whitening"] = whitening_info

    # Preprocessing section
    preprocess_info = {}
    preprocess_info["enabled"] = preprocess
    preprocess_info["mode"] = preprocess_mode
    preprocess_info["datasets"] = [datasets_for_preprocessing] if isinstance(datasets_for_preprocessing, str) else list(datasets_for_preprocessing)

    if preprocess:
        if isinstance(class_for_preprocessing, str):
            preprocess_info["classes"] = {
                class_for_preprocessing: percentage_preprocess
            }
        else:
            preprocess_info["classes"] = {
                cls: percentage_preprocess for cls in class_for_preprocessing
            }

    info["preprocessing"] = preprocess_info

    filepath = os.path.join(save_dir, "dataset_info_extra.json")

    with open(filepath, "w") as f:
        json.dump(info, f, indent=4)

    print(f"Dataset info saved to {filepath}", flush=True)

            
def normalize(x, all_values):
    if x == "all":
        return all_values
    elif isinstance(x, str):
        return (x,)
    else:
        return tuple(x)


def gen_data(classes = ("Clean", "Whistle", "Tomte", "Scattered_Light"), datasets="O3a", 
             whiten=True, class_for_whitening="Clean", datasets_for_whitening="O3a", percentage_whitening=0.9,
             preprocess=True, preprocess_mode = 'standardize', class_for_preprocessing = "Clean", datasets_for_preprocessing="O3a", percentage_preprocess=0.9,
             saving=True, save_dir = '/data/'):
    """Main pipeline function: loads glitch data, optionally computes whitening statistics, whitens the data, 
    preprocesses it, saves parquet datasets and metadata, and exports channel information."""
    
    """for datsets and datasets_for_whitening could put in 'all' for both O3a and O3b data.
    For classes and classes_for_whitening can put in 'all' to use all classes.
    preprocess_mode = 'standardize' or 'normalize'"""
    
    classes_used = normalize(classes, ALL_CLASSES)
    datasets_used = normalize(datasets, ALL_DATASETS)

    # Full dataset
    full_class_names = [
        f"{ds}_L1_{cls}.csv"
        for ds in datasets_used
        for cls in classes_used
    ]


    if whiten:
        print('Whitening data..', flush=True)
        whitening_files = []
        whiten_classes = normalize(class_for_whitening, ALL_CLASSES)
        whiten_datasets = normalize(datasets_for_whitening, ALL_DATASETS)

        whitening_files = [
            f"{ds}_L1_{cls}.csv"
            for ds in whiten_datasets
            for cls in whiten_classes
        ]
        
        _start = time.time()
        data_for_whitening, channel_names = data_from_csv(class_csvs=whitening_files, whiten=False)
        _end = time.time()
        print('Time for generating data for whitening:', _end-_start, flush=True)
        _start = time.time()
        asd_dict, whitened_data_dict = ASD_calc(data=data_for_whitening, percentage_whitening=percentage_whitening, channel_names=channel_names)
        _end = time.time()
        print('Time for calculating PSD:', _end-_start, flush=True)

        classes_left = []
        for class_csv in full_class_names :
            if class_csv not in whitening_files:
                classes_left.append(class_csv)
        whitened_data = None  
        if len(classes_left) > 0:
            # Retrieve data and whiten immidiately
            _start = time.time()
            whitened_data, _ = data_from_csv(class_csvs=classes_left, whiten=True, ASD_dict=asd_dict) 
            _end = time.time()
            print('Time to generate rest of data - immediately whitened:', _end-_start, flush=True)
            whitened_data_dict.update(whitened_data)      # THE WHITENED_DATA DOES NOT HAVE THE CHANNEL NAMES IN ITS DICTIONARY AS SUBKEYS CHECK!
                                                        # MAYBE IT ISN'T USEFUL ANYWAY TO HAVE THE CHANNEL NAMES AS KEYS?

        data_dict = whitened_data_dict

    else:
        print('Generating all data - non-whitening..', flush=True)
        data_dict, channel_names = data_from_csv(class_csvs=full_class_names, whiten=False)
                                                          

    data = 0
    if preprocess:
        print('Preprocessing data..', flush=True)
        _start = time.time()
        # This is a special check to make dataprocessing easier if you use the same data for whitening as for standardizing/normalizing
        if (whiten and class_for_whitening==class_for_preprocessing and datasets_for_whitening==datasets_for_preprocessing and percentage_whitening==percentage_preprocess and whitened_data is not None):
            print('Using same data for preprocessing as for whitening..', flush=True)
            data_prep = whitened_data
            arrays=[]
            for key in list(data_prep.keys()):
                class_data = data_prep[key]
                if isinstance(class_data, dict):
                    arrays.append(class_data['train'])
                else:
                    arrays.append(class_data)
            data = np.concatenate(arrays, axis=0)

            final_data = {}

            if preprocess_mode == "standardize":
                print('Standardizing', flush=True)
                mean = np.mean(data, axis=(0,1))
                std = np.std(data, axis=(0,1))

                for class_name in full_class_names:
                    class_name = class_name[:-4]
                    class_data = data_dict[class_name]
                    
                    if isinstance(class_data, dict):
                        train = (class_data['train'] - mean) / std
                        test = (class_data['test'] - mean) / std
                        
                        final_data[class_name] = {}  # initialize the dict first
                        final_data[class_name]['train'] = train
                        final_data[class_name]['test'] = test
                    else:
                        full = (class_data - mean) / std
                        final_data[class_name] = full
    
            if preprocess_mode == 'normalize':
                print('Normalizing', flush=True)
                min = np.min(data, axis=(0,1))
                max = np.max(data, axis=(0,1))

                for class_name in full_class_names:
                    class_name = class_name[:-4]
                    class_data = data_dict[class_name]
                    if isinstance(class_data, dict):
                        train = (class_data['train'] - min) / (max - min)
                        test = (class_data['test'] - min) / (max - min)

                        final_data[class_name] = {}  # initialize the dict first
                        final_data[class_name]['train'] = train
                        final_data[class_name]['test'] = test
                    else:
                        full = (class_data - min) / (max - min)
                        final_data[class_name] = full

            
        else:
            preprocess_class = []
            prep_classes = normalize(class_for_preprocessing, ALL_CLASSES)
            prep_datasets = normalize(datasets_for_preprocessing, ALL_DATASETS)

            preprocess_class = [
                f"{ds}_L1_{cls}"
                for ds in prep_datasets
                for cls in prep_classes
            ]
        
        
            final_data = preprocess_data(data=data_dict, classes=preprocess_class, percentage_preprocess=percentage_preprocess, mode=preprocess_mode)

        _end = time.time()
        print('Time for preprocessing:', _end-_start, flush=True)
    else:
        final_data = data_dict

    

    # ===== Saving data to Parquet =====
    if saving:
        
        # ===== Save channel names =====
        channel_txt_path = os.path.join(save_dir, "channel_names.txt")

        with open(channel_txt_path, "w") as f:
            for ch in channel_names:
                f.write(f"{ch}\n")

        print(f"Channel names saved to: {channel_txt_path}", flush=True)

        save_dir=save_dir
        final_classes = list(final_data.keys())
        
        for class_name in final_classes:
            print('Class_name:', class_name, flush=True)
            class_data = final_data[class_name]

            if isinstance(class_data, dict):
                train_data = class_data['train']
                test_data = class_data['test']
                train_samples,n_timesteps,_ = train_data.shape
                test_samples,_,_ = test_data.shape

                df_train = pd.DataFrame(train_data.reshape(train_samples*n_timesteps, -1), columns = channel_names).dropna(axis=1, how='all')
                df_test = pd.DataFrame(test_data.reshape(test_samples*n_timesteps, -1), columns = channel_names).dropna(axis=1, how='all')

                path_train = save_dir + f"{class_name}_train.parquet"
                path_test = save_dir + f"{class_name}_test.parquet"
                df_train.to_parquet(path=path_train)
                df_test.to_parquet(path=path_test)
            else:
                n_samples,n_timesteps,_ = class_data.shape
                full_df = pd.DataFrame(class_data.reshape(n_samples*n_timesteps, -1), columns = channel_names).dropna(axis=1, how='all')
                path = save_dir + f"{class_name}_extra.parquet"
                full_df.to_parquet(path=path)


    save_dataset_info(save_dir=save_dir,
                      classes=classes,
                      datasets=datasets,
                      whiten=whiten,
                      class_for_whitening=class_for_whitening,
                      datasets_for_whitening=datasets_for_whitening,
                      percentage_whitening=percentage_whitening,
                      preprocess=preprocess,
                      preprocess_mode=preprocess_mode,
                      class_for_preprocessing=class_for_preprocessing,
                      datasets_for_preprocessing=datasets_for_preprocessing,
                      percentage_preprocess=percentage_preprocess)
    print('Done!', flush=True)

    return 


gen_data(classes = ("Tomte", "Whistle", "Scattered_Light"), datasets="O3a", whiten=False, class_for_whitening=("Clean"), datasets_for_whitening="O3a", percentage_whitening=(0.7), preprocess=False, preprocess_mode = 'standardize', class_for_preprocessing = "Clean", datasets_for_preprocessing="O3a", percentage_preprocess=(0.7), saving=True)
