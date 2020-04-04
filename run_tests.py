import learn_model as lm
import sys
import os
import pickle as pl
from datetime import date
import plot_results as pr


def blockPrint():
    sys.__stdout__ = sys.stdout
    sys.stdout = open(os.devnull, 'w')
def enablePrint():
    sys.stdout = sys.__stdout__


batch_sizes = [256, 128, 64, 32]


def run_tests_train():
    n_epochs = 200
    patience = 50
    fold = 1
    use_embed_layer = 'True'

    blockPrint()

    results = []
    for task in batch_sizes: results.append(lm.execute(fold, task, n_epochs, patience, use_embed_layer))

    enablePrint()
    for r in results: print(r, '\n')

    # save the results
    today = date.today()
    if use_embed_layer:
        experiment = ' with aux net results'
    else:
        experiment = ' without aux net results'

    res_file_name = str(today) + experiment
    with open(res_file_name + '.pkl', 'wb') as f:
        pl.dump(results, f)
    print('Saved results data file: ', res_file_name + '.pkl')

    # [0. train_losses, 1. valid_losses, 2. train_accs, 3. valid_accs, 4. test_acc, 5. epoch_times, 6. train_time]
    train_accs = list(r[2] for r in results)
    valid_accs = list(r[3] for r in results)
    test_accs = list(r[4] for r in results)

    # visualize the loss
    pr.plot_results(experiment, valid_accs, batch_sizes, test_accs)


if __name__ == '__main__':
    file_name = '2020-04-04 without aux net results.pkl'
    experiment = 'epoch_times without aux net'

    with open(file_name, 'rb') as f:
        results = pl.load(f)

    train_accs = list(r[2] for r in results)
    valid_accs = list(r[3] for r in results)
    test_accs = list(r[4] for r in results)

    train_losses = list(r[0] for r in results)
    valid_losses = list(r[1] for r in results)
    epoch_times = list(r[5] for r in results)
    train_time = list(r[6] for r in results)

    pr.plot_results(experiment, epoch_times, batch_sizes, train_time)
