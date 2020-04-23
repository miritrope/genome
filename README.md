# Genomic classification using neural networks 
Pytorch implementation of neural networks for genetic tests classification, inspired by the paper: [DietNetworks](https://arxiv.org/abs/1611.09340/).
For further explanation please read [my blog](https://medium.com/@miritrope_17849/bridging-the-gap-between-genetics-and-neural-networks-37563709537d): "Bridging the gap between genetics and neural networks".
 
## Usage
Step 1: Follow the instructions in appendix B of the paper to download the data 

Step 2: Generate the embedding matrix (optional)
` python utils_helpers.py`

Step 3: Train the network 
` python run_tests.py -file_name your_file_name.pkl -n_epochs 1000 -use_embed_layer 0  -fold 1 -patience 100 -hidden_sizes 50 -dropout_1 0.8 -dropout_2 0.5 `

Parameters:

|  Name | Required | Type  | Description
| ----- | -------- |------ |----------- |
| file_name      | Required | str  | bases and labels  |
| n_epochs       | Required | int  | number of epochs  |
| use_embed_layer| Required | int  | whether to use the auxiliary network  |
| fold           | Optional | int  | which fold of the embedding layer dataset  |
| patience       | Optional | int  | patience of the early stopping  |
| hidden_sizes   | Required | int  | hidden layer units sizes  |
| dropout_1      | Required | float| dropout of the first hidden layer  |
| dropout_2      | Required | float| dropout of the second hidden layer |


### Requirements
must:
* Torch
* Numpy
* Plink

optional:
* Matplotlib
* Pickle
