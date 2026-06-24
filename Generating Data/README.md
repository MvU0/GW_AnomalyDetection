# Generating auxiliary channel time series data sets

The function gen_data() generates time series data sets seperated by class, and saves them as .parquet files in the shape[N_samples*N_timesteps, channels]

Of each channel 8 seconds are used and are downsampled to 256 Hz, thus N_timesteps = 2048.


# Settings
* classes = 
    Which classes you want to generate time series data for
    Example: ("Clean", "Whistle", "Tomte", "Scattered_Light")
* datasets =
    Either "O3a", "O3b" or "all" to get both data sets.
* whiten =
    Boolean to say if data needs to be whitened
* class_for_whitening =
    Which class to use to calculate PSD used for whitening
    Examples: "("Clean", "Whistle",), ("Clean",) or "all"
* datasets_for_whitening = 
    Which data set to use for whitening
    Either "O3a", "O3b" or "all"
* percentage_whitening =
    Percentage of specified whitening data to use for calculating the PSD
* preprocess=True
    Boolean to say if data needs to be preprocessed
* preprocess_mode = 
    Method of perprocessing.
    Either 'standardize' or 'normalize'
* class_for_preprocessing = 
    Which data set to use for preprocessing
    Examples: "("Clean", "Whistle",), ("Clean",) or "all"
* datasets_for_preprocessing =
    Which data set to use for prerpocessing
    Either "O3a", "O3b" or "all"
* percentage_preprocess =
    Percentage of specified preprocessing data to use
* saving = 
    Boolean to save data
* save_dir =
    Directory where to save generated data sets
    Example: /data/


# Output
* Outputs a config file containing all settings
* Outputs data sets per class