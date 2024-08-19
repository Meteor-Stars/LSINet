# LSINet

Our experimental environments involve Pytorch 1.12.1 and Numpy 1.22.4. 


## Downloading Datasets
  You can download the public datasets used in our paper from https://drive.google.com/drive/folders/1PPLsAoDbv4WcoXDp-mm4LFxoKwewnKxX. The downloaded files e.g., "ETTh1.csv",  should be placed at the "dataset" folder. These datasets are extensively used for evaluating performance of various time series forecasting methods.
  
## Reproducing Paper Results
We have provided the experimental run scripts for LSINet and baseline models (CI-TSmixer, FiLM, DLinear, PatchTST, TimeMixer, Scaleformer, Pathformer, and FiLM) on the public datasets. The corresponding model names and forecasting tasks are included in the script names. 
The hyperparameters used for experiments of different methods have been set in their respective scripts to reproduce the experimental results of the paper, e.g., running the script "Run_LSINet_TSF (Ours).py" can reprpduce the paper results of our LSINet.
  
