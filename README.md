# Genome classification using neural network in Pytorch
Implementation of neural network for genetic tests classification, inspired by the experiment of the paper [DietNetworks:](https://arxiv.org/abs/1611.09340/) Thin parameters for fat genomic.

## Usage
Step 1: Follow the instructions in appendix B of the paper to download the data.

Step 2: Generate the embedding matrix
` python utils_helpers.py`

Step 3: Train the network 
  ``python learn_model.py``
  
  
### Requirements
1. torch
2. numpy
3. matplotlib
4. pickle
