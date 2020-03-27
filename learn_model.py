#!/usr/bin/env python
import os
import argparse

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

from pytorchtools import EarlyStopping
import mainloop_helpers as mlh
import model_helpers as mh
import time

_EPSILON = 10e-8


def execute():
    fold = 0
    embedding_source = 'histo3x26_fold'
    raw_path = 'affy_6_biallelic_snps_maf005_thinned_aut_dataset.pkl'
    # dataset_path = '/Users/miri/PycharmProjects/genome/data/'
    dataset_path = 'data/'
    batch_size = 80
    learning_rate = 3e-5
    n_hidden_u = 100
    n_hidden_t_enc = 100
    encoder_net_init = 0.02
    disc_nonlinearity = "softmax"
    batchnorm = True
    n_targets = 26
    n_hidden_s = 100
    lmd = .0001
    num_epochs = 1000
    batch_size = 80
    patience = 50


    print("Loading data")
    x_train, y_train, x_valid, y_valid, x_test, y_test, \
    x_unsup, training_labels = mlh.load_data(dataset_path, raw_path, embedding_source, fold)

    print('declares shared veriables')
    feat_emb = Variable(torch.from_numpy(x_unsup), requires_grad=True)
    #feat_emb shape: 315 x 104
    n_feats = feat_emb.shape[1]


    print("Building embedding model")
    emb_model = mh.feat_emb_net(n_feats, n_hidden_u, n_hidden_t_enc)
    embedding = emb_model(feat_emb)
    # embedding size is:315 x 100

    # transpose to fit the weights in discriminative network
    embedding = torch.transpose(embedding,1,0)


    print("Building discrim model")
    discrim_model = mh.discrim_net(embedding, feat_emb.shape[0], n_hidden_u, n_hidden_t_enc, n_targets)
    print("Done building discrim model")

    # input_discrim = Variable(torch.randn(batch_size, feat_emb.shape[0]).type(dtype), requires_grad=False)
    # input_discrim size = 80,315
    # y_pred = discrim_model(input_discrim)


    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(discrim_model.parameters(), lr=learning_rate)


    # Apply norm constraints on the weights
    # for k in updates.keys():
    #     if updates[k].ndim == 2:
    #         updates[k] = lasagne.updates.norm_constraint(updates[k], 1.0)


    # Finally, launch the training loop.
    print("Starting training...")


    train_minibatches = list(mlh.iterate_minibatches(x_train, y_train,
                                                     batch_size))

    valid_minibatches = list(mlh.iterate_minibatches(x_valid, y_valid,
                                                    batch_size))

    test_minibatches = list(mlh.iterate_minibatches(x_test, y_test,
                                                    batch_size))



    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)


    train_step = mh.make_train_step(discrim_model, loss_fn, optimizer)


    epoch_len = len(str(num_epochs))

    train_losses_per_batch = []
    valid_losses_per_batch = []
    test_losses_per_batch = []
    for epoch in range(num_epochs):
        print("Epoch {} of {}".format(epoch + 1, num_epochs))


        train_batch_losses = []
        valid_batch_losses = []

        for x_batch, y_batch in train_minibatches:
            x_train = Variable(torch.from_numpy(x_batch))
            y_train = Variable(torch.from_numpy(y_batch))

            loss = train_step(x_train, y_train)
            #vector of losses per batch
            train_batch_losses.append(loss)

        train_losses_per_batch.append(np.average(train_batch_losses))

        discrim_model.eval()

        with torch.no_grad():
            for x_val, y_val in valid_minibatches:
                x_val = Variable(torch.from_numpy(x_val))
                y_val = Variable(torch.from_numpy(y_val))

                yhat = discrim_model(x_val)
                loss = loss_fn(y_val, yhat)
                valid_batch_losses.append(loss.item())

            valid_losses_per_batch.append(np.average(valid_batch_losses))


        #finished a batch
        print_msg = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_losses_per_batch[epoch]:.5f} ' +
                     f'valid_loss: {valid_losses_per_batch[epoch]:.5f}')


        print(print_msg)


        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_losses_per_batch[epoch], discrim_model)

        if early_stopping.early_stop:
            print("Early stopping")
            break




    # load the last checkpoint with the best model
    discrim_model.load_state_dict(torch.load('checkpoint.pt'))


    # visualize the loss
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_losses_per_batch) + 1), train_losses_per_batch, label='Training Loss')
    plt.plot(range(1, len(valid_losses_per_batch) + 1), valid_losses_per_batch, label='Validation Loss')

    #find position of lowest validation loss
    minposs = valid_losses_per_batch.index(min(valid_losses_per_batch)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')


    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim(0, len(train_losses_per_batch) + 1)  # consistent scale
    plt.ylim(0, 0.04)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('loss_plot.png', bbox_inches='tight')


    # test
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(n_targets))
    class_total = list(0. for i in range(n_targets))

    discrim_model.eval()

    # test_batch_losses = []

    for x_test, y_test in test_minibatches:
        x_test = Variable(torch.from_numpy(x_test))
        y_test = Variable(torch.from_numpy(y_test))

        yhat = discrim_model(x_test)
        loss = loss_fn(y_test, yhat)

        test_loss += loss.item() * batch_size

        # test_batch_losses.append(loss.item())

        # convert output probabilities to predicted class
        _, pred = torch.max(yhat, 1)
        # compare predictions to true label
        y_test = np.argmax(y_test.data, axis=1)

        correct = np.squeeze(pred.eq(y_test.view_as(pred)))
        #correct size: [80]
        # calculate test accuracy for each object class
        y_test = y_test.tolist()

        for i in range(batch_size):
            label = y_test[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculate and print avg test loss
    test_loss = test_loss / len(test_minibatches)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(n_targets):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))



    print('Training End')


def main():
    parser = argparse.ArgumentParser(description="""Train Genome""")
    parser.add_argument('--embedding_source',
                        default='histo3x26',
                        help='Source for the feature embedding')
    parser.add_argument('--dataset_path',
                        default='/Users/miri/PycharmProjects/genome/data',
                        help='Path to dataset')

    args = parser.parse_args()

    execute()


if __name__ == '__main__':
    main()
