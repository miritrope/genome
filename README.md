# Genomic classification using neural networks 
Pytorch implementation of neural networks for genetic tests classification, inspired by the experiment of the paper "[DietNetworks:](https://arxiv.org/abs/1611.09340/) Thin parameters for fat genomic".

## Usage
Step 1: Follow the instructions in appendix B of the paper to download the data.

Step 2: Generate the embedding matrix
` python utils_helpers.py`

Step 3: Train the network 
  ``python learn_model.py``
  
  
### Requirements
must:
* Torch
* Numpy
* Plink

optional:
* Matplotlib (to view the graph of the loss function)
* Pickle (to load the genomic data)
