# Auxiliary Classifier
Model designed to use LIGO-L1 auxiliary channel data to classify 8 second segments as normal or anomalous data
The model include two methods of determining channel importance in glitch classification as explained in the thesis: 

### Start
Works with both GPU and CPU (although a lot faster on GPU of course)
---DATA
* Create a time series data set containing all data - shaped [N_samples, N_timesteps, N_channels] and save as .npy file
* Create a set of labels - shaped [N_samples] and save as .npy file
* If other data type or convention is used change the data loading. The model requires input in the shape of [N_samples, x, y] where x,y may be     either N_timesteps, N_channels or N_channels, N_timesteps. 
* Data can be found at stoomboot at: /data/gravwav/mvuden/anomaly-detection-thesis-mees-van-uden/New_GNN_start/_Dataset_classifier/Timeseries/GAT/Optuna/combined_data_NTWS.npy
* Labels can be found at: /data/gravwav/mvuden/anomaly-detection-thesis-mees-van-uden/New_GNN_start/_Dataset_classifier/Timeseries/GAT/Optuna/combined_labels_NTWS.npy

* Create a list of channel names corresponding to the data set

* It should look like:
Classifier model
 |-GW
 |-GAT_classifier.ipynb
 |-classification.py    
 |-your_dataset.npy
 |-your_labels.npy    
 |-channel_names.txt
 | ...


 ### Settings
 Settings are explained at the start of the classification.py file
 In GAT_classifier.ipynb only the GAT_classifier is used, however it is straightforward to go to either the CNN or MLP classifier defined in classification.py


 ### Output
 GAT_classifier.ipynb outputs the training process and contains code to check channel importance in two manners. One for when nodes represent time series and one for when nodes represent channels. 
