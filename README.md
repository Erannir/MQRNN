# MQRNN

This is a partial implementation to the paper [A Multi-Horizon Quantile Recurrent Forecaster](https://arxiv.org/abs/1711.11053).

## Files

* ElectricityLoadDataset.py - a torch.utils.dataset object for the a dataset of electricy consumption by 370 households. 
* Models.py - implementation of the MQRNN achitecture from the paper, including:
  * Encoder-Decoder achitecture using LSTM-MLPs design.
  * Sequence Forking for better utialization of training data.
  * Efficient implementation of the decoder, including stacking tensors for all t\inT, k\inK (see paper) for a single forward use of MLPs (no loops), and single backprop through time of entire NN. 
* loss_functions.py - implementation of the quantile-regression loss function from the paper.
* plot_utils.py - self explanatory.
* training.py - training code, including an Estimator class used for both training and inference. Recommended for using on PC.
* MQRNN.ipynb - a wrapper for training.py, for running in the cloud.

## Setup
Please be sure to have the following folders under your code folder:
* data - including the data file provided in class. If data is changed, please edit ElectricityLoadDataset.py 
* models - for saving the model parameters during runtime (done in valid eval step). One pretrained model is supplied for inference without training.
* graphs - if you chose to save the graphs created in Estimator.forecast().

## Submitted by
Tomer Koren
Tzika Duetsch
Eran Nir
