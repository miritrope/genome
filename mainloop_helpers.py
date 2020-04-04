import numpy as np
import dataset_utils as du
import torch


def load_data(data_path, raw_path, emb_path, fold):
    # norm by default is true because for training the samples is normalized
    data = du.load_1000_genomes(data_path, raw_path, fold=0, norm=True)
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = data

    feat_emb_val = du.load_embedding_mat(data_path, emb_path, fold=0, transpose=False)

    training_labels = y_train

    return x_train, y_train, x_valid, y_valid, x_test, y_test, \
           feat_emb_val, training_labels


# Mini-batch iterator function
def iterate_minibatches(inputs, targets, batchsize):
    assert inputs.shape[0] == targets.shape[0]
    indices = np.arange(inputs.shape[0])

    for i in range(0, inputs.shape[0]-batchsize+1, batchsize):
        yield inputs[indices[i:i+batchsize], :],\
            targets[indices[i:i+batchsize]]


def make_train_step(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def train_step(x, y):
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        # input_discrim size = 80,315
        yhat = model(x)
        _, pred = torch.max(yhat, 1)

        # Computes loss
        loss = loss_fn(y, yhat)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item(), pred

    # Returns the function that will be called inside the train loop
    return train_step


if __name__ == '__main__':
    x_train, y_train, x_valid, y_valid, x_test, y_test, \
    feat_emb_val, training_labels = load_data(data_path ='data/', raw_path='affy_6_biallelic_snps_maf005_thinned_aut_dataset.pkl', emb_path='histo3x26_fold', fold=0)

    iterate_minibatches(x_train, y_train, batchsize=80)
    print("Loaded data")
