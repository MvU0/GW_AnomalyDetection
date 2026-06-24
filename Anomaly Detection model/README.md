# GDN

Original code implementation for : [Graph Neural Network-Based Anomaly Detection in Multivariate Time Series(AAAI'21)](https://arxiv.org/pdf/2106.06947.pdf)
Adapted to auxiliary LIGO-L1 data. Used for glitch detection and search for witness channels.


# Installation
### Requirements
* Python >= 3.6
* cuda == 10.2
* [Pytorch==1.5.1](https://pytorch.org/)
* [PyG: torch-geometric==1.5.0](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

### Install packages
```
    # run after installing correct Pytorch package
    bash install.sh
```

### Start
To be used with HTCondor on GPU
---DATA
* Create a folder within data folder with name of dataset used (example: GW or OCEAN)
* Put normal and anomalous data in data folder as .parquet files (which is an effective pandas dataframe method)
* Name the files: normal_data.parquet / anomaly_data.parquet  -- When using different naming convention change lines 55 and 56 in main.py
* It should look like:
data
 |-GW
 | |-normal_data.parquet   
 | |-anomaly_data.parquet     
 |-your_dataset
 | |-normal_data.parquet   
 | |-anomaly_data.parquet  
 | ...



---Settings
* In main.sub change settings of model
* If different datasets are used important setting is DATASET = 
* All settings are passed on to the main.py file
* Subsystems selects which subsystems of the LIGO detector to use. Can use multiple subsystems like: Subsystems = IMC, SUS, SQZ
        Can also use Subsystems = FULL -- To get all channels

---Submitting
* To submit run the following line in a terminal when in the right directory
```
    condor_submit main.sub
```
* Need to adjust main.sh to activate the right environment and point to the right directory

---Saving
* Continuous error scores per channel for validation and test data are saved in_analysis/err_scores
* Evaluation scores and threshold are saved in results/your_dataset/{datestr + settings}/time_window.csv
* Loss and logloss are plotted in results/your_dataset/imgs
* Predicted vs true values per time step of test data is saved in _analysis/values
* Predicted vs true labels per time step of test data is saved in _analysis/your_dataset/{settings}/time_window.csv
* Graph is saved in _analysis/graph/{settings}.pt
* Node embeddings are saved in embeddings/specifics   -- the embeddings can be viewed using tSNE and interactive local host - see embeddings/HowToOpen.md


### Analysis
* Analysis code in _analysis/
* Two codes for analysis: 
        1. confusion_matrix.ipynb -- Visualises classification and shows which channels cause classification per glitch class
        2. predictions_vs_real.ipynb -- Visualises the forecasted and true values in the model and allows direct visual comparison across different classes.

### Data
* Data is found in /data/gravwav/mvuden/anomaly-detection-thesis-mees-van-uden/GNN_rework/data/GW/
    -- Normal data is saved under: normal_data.parquet ; glitch data is saved under anomaly_data.parquet
       This is raw time series data. In the current state of the code all data is standardized by all normal_data. Standardization happens in util/data/generate_datasets()

    -- Extended datasets are saved in same directory under: normal_data_extended.parquet and anomaly_data_extended.parquet
            however extended normal data performs significantly worse, which may be due to it coming from different part of observing run

* Needs to be careful if data does not contain duplicates.

### Notices:
* The current setup for test data makes assumptions about how anomaly_data is structured. It assumes it contains an equal amount of three types of glitches. See util/data/generate_datasets() for how anomalous data is treated

