import learn_model as lm
import sys
import os
import pickle as pl
import plot_results as pr


def blockPrint():
    sys.__stdout__ = sys.stdout
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    sys.stdout = sys.__stdout__


def run_results(results, exp_name, n_epochs):

    # extract results data
    train_losses = results[0]
    valid_losses = results[1]
    train_accs = results[2]
    valid_accs = results[3]
    test_acc = results[4]
    epoch_times = results[5]
    train_time = results[6]

    res_file_name = exp_name + ' test acc ' + f'{test_acc:.1f}'

    # save the results
    with open(res_file_name + '.pkl', 'wb') as f:
        pl.dump(results, f)
    print('Saved results data file: ', res_file_name + '.pkl')

    pic_name = res_file_name + '.png'
    pr.plot_results([train_losses, valid_losses],
                    n_epochs, pic_name)


if __name__ == '__main__':

    patience = 50
    fold = 1
    use_embed_layer = False
    n_epochs = 3000
    tasks = [[[50, 50],   [80, 80],   [50, 50]],
        [[0.8, 0.8], [0.8, 0.2], [0.8, 0.2]]]

    for i in range(len(tasks[0])):

        # define experiment name
        experiment = ''
        if use_embed_layer:
            experiment += 'with '
        else:
            experiment += 'without '

        n_hidden = tasks[0][i]
        drop_sizes = tasks[1][i]

        experiment += ('ep ' + str(n_epochs) +
                        ' hi ' + str(n_hidden) +
                            ' dr ' + str(drop_sizes))

        print(experiment)

        results = lm.execute(fold, n_hidden, n_epochs,
                   patience, use_embed_layer, drop_sizes)

        run_results(results, experiment, n_epochs)
