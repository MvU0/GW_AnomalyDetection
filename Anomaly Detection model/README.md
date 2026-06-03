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
* Put fully preprocessed train/val/test data in data folder as .parquet files (which is an effective pandas dataframe method)
* Name the files: train.parquet / val.parquet / test.parquet  -- When using different naming convention change lines 53, 54, and 83 in main.py
* It should look like:
data
 |-GW
 | |-list.txt        # the feature names, one feature per line
 | |-train.parquet   # training data
 | |-val.parquet     # validation data
 | |-test.parquet    # test data
 |-your_dataset
 | |-list.txt
 | |-train.parquet
 | |-val.parquet
 | |-test.parquet
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
---Saving
* Continuous error scores per channel for validation and test data are saved in_analysis/err_scores
* Evaluation scores and threshold are saved in results/your_dataset/{datestr + settings}/time_window.csv
* Loss and logloss are plotted in results/your_dataset/imgs
* Predicted vs true values per time step of test data is saved in _analysis/values
* Predicted vs true labels per time step of test data is saved in _analysis/your_dataset/{settings}/time_window.csv
* Graph is saved in _analysis/graph/{settings}.pt
* Node embeddings are saved in embeddings/specifics   -- the embeddings can be viewed using tSNE and interactive local host - see HowToOpen.md




### Notices:
* The column sequence in .parquet don't need to match the sequence in list.txt, we will rearrange the data columns according to the sequence in list.txt.
* test.csv should have a column named "attack" which contains ground truth label(0/1) of being attacked or not(0: normal, 1: attacked)


