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

_EPSILON = 10e-8


def execute():
    fold = 0
    embedding_source = 'embed_4x26_fold'
    raw_path = 'affy_6_biallelic_snps_maf005_thinned_aut_dataset.pkl'
    dataset_path = 'data/'
    batch_size = 80
    learning_rate = 3e-5
    n_hidden_u = 100
    n_hidden_t_enc = 100
    n_targets = 26
    num_epochs = 500
    patience = 30

    print("Load data")
    x_train, y_train, x_valid, y_valid, x_test, y_test, \
    x_unsup, training_labels = mlh.load_data(dataset_path, raw_path, embedding_source, fold)

    # Declare shared veriables
    feat_emb = Variable(torch.from_numpy(x_unsup), requires_grad=True)
    # feat_emb size: n_feats x 104
    n_feats = feat_emb.shape[1]

    print('Build models')
    # Build embedding model
    emb_model = mh.feat_emb_net(n_feats, n_hidden_u, n_hidden_t_enc)
    embedding = emb_model(feat_emb)
    # embedding size: n_feats x n_hidden_t_enc

    # transpose to fit the weights in the discriminative network
    embedding = torch.transpose(embedding, 1, 0)

    # Build discrim model
    discrim_model = mh.discrim_net(embedding, feat_emb.shape[0], n_hidden_u, n_hidden_t_enc, n_targets)

    # some comments:
    # input_discrim size: batch_size, n_feats
    # input_discrim = Variable(torch.randn(batch_size, feat_emb.shape[0]).type(dtype), requires_grad=False)
    # y_pred = discrim_model(input_discrim)

    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(discrim_model.parameters(), lr=learning_rate)

    # TODO: Apply norm constraints on the weights
    # for k in updates.keys():
    #     if updates[k].ndim == 2:
    #         updates[k] = lasagne.updates.norm_constraint(updates[k], 1.0)

    # Finally, launch the training loop.
    print("Start training ...")
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

    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):
        print("Epoch {} of {}".format(epoch + 1, num_epochs))

        train_loss = 0.0
        for x_batch, y_batch in train_minibatches:
            x_train = Variable(torch.from_numpy(x_batch))
            y_train = Variable(torch.from_numpy(y_batch))

            loss = train_step(x_train, y_train)
            train_loss += loss * batch_size

        discrim_model.eval()

        valid_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in valid_minibatches:
                x_val = Variable(torch.from_numpy(x_val))
                y_val = Variable(torch.from_numpy(y_val))

                yhat = discrim_model(x_val)
                loss = loss_fn(y_val, yhat)

                valid_loss += loss.item() * batch_size

        # finished a batch

        train_loss = train_loss / len(train_minibatches)
        valid_loss = valid_loss / len(valid_minibatches)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print_msg = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, discrim_model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    discrim_model.load_state_dict(torch.load('checkpoint.pt'))

    # visualize the loss
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_losses.index(min(valid_losses)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim(0, len(train_losses) + 1)  # consistent scale
    plt.ylim(0, 4)  # consistent scale
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

    for x_test, y_test in test_minibatches:
        x_test = Variable(torch.from_numpy(x_test))
        y_test = Variable(torch.from_numpy(y_test))

        yhat = discrim_model(x_test)
        loss = loss_fn(y_test, yhat)

        test_loss += loss.item() * batch_size

        # convert output probabilities to predicted class
        _, pred = torch.max(yhat, 1)
        # compare predictions to true label
        y_test = np.argmax(y_test.data, axis=1)

        # correct size: [batch_size]
        correct = np.squeeze(pred.eq(y_test.view_as(pred)))
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
