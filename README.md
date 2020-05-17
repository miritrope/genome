# Genetics Classification Using Neural Networks 
_Pytorch_ Implementation for Genetics Classification. 
For further explanation please read my blog: [Bridging the Gap Between Genetics and Neural Networks](https://towardsdatascience.com/bridging-the-gap-between-genetics-and-neural-networks-4fdb91032f4b)
 
## Usage

Step 1: Generate the Embedding Matrix (optional)
` python utils_helpers.py`

Step 2: Train the Network 
` python run_tests.py -file_name your_file_name.pkl -batch_size 64 -n_epochs 1000 -use_embed_layer 0  -fold 1 -patience 100 -hidden_sizes 50 -dropout_1 0.8 -dropout_2 0.5 `

Parameters

|  Name | Required | Type  | Description
| ----- | -------- |------ |----------- |
| file_name      | required | str  | bases and labels  |
| batch_size     | required | int  | batch sizes [32, 64, 128, 256] |
| n_epochs       | required | int  | number of epochs  |
| use_embed_layer| required | int  | using the auxiliary network  |
| fold           | optional | int  | fold of the embedding dataset  |
| patience       | optional | int  | patience of the early stopping  |
| hidden_sizes   | required | int  | hidden layer unit sizes  |
| dropout_1      | required | float| dropout of the first hidden layer  |
| dropout_2      | required | float| dropout of the second hidden layer |


### Requirements
Must
* Torch
* Numpy

Optional
* Matplotlib
* Pickle
