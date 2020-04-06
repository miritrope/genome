#!/usr/bin/env python
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from pytorchtools import EarlyStopping
import mainloop_helpers as mlh
import model_helpers as mh
import time


_EPSILON = 10e-8


def execute(fold, batch_size, n_epochs, patience, use_embed_layer):
    raw_path = 'affy_6_biallelic_snps_maf005_thinned_aut_dataset.pkl'
    dataset_path = 'data/'

    embedding_source = []
    if use_embed_layer:
        embedding_source = 'embed_4x26_fold'

    learning_rate = 3e-5
    n_hidden_1 = 50
    n_hidden_2 = 50
    n_targets = 26

    # print("Load data")
    x_train, y_train, x_valid, y_valid, x_test, y_test, \
    x_unsup, training_labels = mlh.load_data(dataset_path, raw_path, embedding_source, fold)
    n_feats = x_train.shape[1]

    print('Build models')
    # Build discrim model
    if use_embed_layer:
        n_emb = x_unsup.shape[1]
        feat_emb = Variable(torch.from_numpy(x_unsup))
        # Build embedding model
        emb_model = mh.feat_emb_net(n_emb, n_hidden_1, n_hidden_2)
        embedding = emb_model(feat_emb)
        # embedding size: n_emb x n_hidden_2
        # transpose to fit the weights in the discriminative network
        embedding = torch.transpose(embedding, 1, 0)
        discrim_model = mh.discrim_net(embedding, n_feats, n_hidden_1, n_hidden_2, n_targets)

    else:
        discrim_model = mh.discrim_net([], n_feats, n_hidden_1, n_hidden_2, n_targets)

    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(discrim_model.parameters(), lr=learning_rate)

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
    train_step = mlh.make_train_step(discrim_model, loss_fn, optimizer)
    epoch_len = len(str(n_epochs))

    train_losses = []
    valid_losses = []

    valid_accs = []
    train_accs = []
    start_training = time.time()
    epoch_times = []
    for epoch in range(n_epochs):
        print("Epoch {} of {}".format(epoch + 1, n_epochs))
        epoch_start_time = time.time()

        class_correct = list(0. for i in range(n_targets))
        class_total = list(0. for i in range(n_targets))
        train_loss = 0.
        for x_batch, y_batch in train_minibatches:
            x_train = Variable(torch.from_numpy(x_batch))
            y_train = Variable(torch.from_numpy(y_batch))

            loss, pred = train_step(x_train, y_train)
            train_loss += loss * batch_size

            # compare predictions to true label
            y_train = np.argmax(y_train.data, axis=1)
            correct = np.squeeze(pred.eq(y_train.view_as(pred)))

            for i in range(batch_size):
                label = y_train[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

        train_acc = 100. * np.sum(class_correct) / np.sum(class_total)
        train_accs.append(train_acc)

        discrim_model.eval()

        valid_loss = 0.
        with torch.no_grad():
            for x_val, y_val in valid_minibatches:
                x_val = Variable(torch.from_numpy(x_val))
                y_val = Variable(torch.from_numpy(y_val))

                yhat = discrim_model(x_val)
                loss = loss_fn(y_val, yhat)

                valid_loss += loss.item() * batch_size

                # compare predictions to true label
                _, pred = torch.max(yhat, 1)
                y_val = np.argmax(y_val.data, axis=1)
                correct = np.squeeze(pred.eq(y_val.view_as(pred)))

                for i in range(batch_size):
                    label = y_val[i]
                    class_correct[label] += correct[i].item()
                    class_total[label] += 1

            valid_acc = 100. * np.sum(class_correct) / np.sum(class_total)
            valid_accs.append(valid_acc)


        # finished a batch

        train_loss = train_loss / len(train_minibatches)
        valid_loss = valid_loss / len(valid_minibatches)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print_msg = (f'[{epoch+1:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg)

        epoch_time = time.time() - epoch_start_time
        # print("epoch time: {:.3f}s".format(epoch_time))
        epoch_times.append(epoch_time)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, discrim_model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    discrim_model.load_state_dict(torch.load('checkpoint.pt'))

    # test
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.
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

    # printing results
    test_loss = test_loss / len(test_minibatches)
    print('Test Loss: {:.6f}\t'.format(test_loss))

    train_time = time.time() - start_training
    test_acc = 100. * np.sum(class_correct) / np.sum(class_total)
    print('Test Accuracy (Overall): %2f%% (%2d/%2d)\t' % (
        test_acc, np.sum(class_correct), np.sum(class_total)))

    return [train_losses, valid_losses, train_accs, valid_accs, test_acc, epoch_times, train_time]


def main():
    execute(fold, batch_size, n_epochs, patience, use_embed_layer)


if __name__ == '__main__':
    n_epochs = 2
    patience = 50
    fold = 1
    execute(fold, 128, n_epochs, patience, False)
