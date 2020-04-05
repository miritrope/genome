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


def test_train():
    # batch_sizes = [256, 128, 64, 32]
    batch_sizes = [64]

    n_epochs = 3000
    patience = 50
    fold = 1
    use_embed_layer = True
    n_experiments = 1

    print('Training ... ')
    blockPrint()

    results = []
    for i in range(n_experiments):
        for task in batch_sizes: results.append(lm.execute(fold, task, n_epochs, patience, use_embed_layer))

    enablePrint()
    for r in results: print(r, '\n')

    # save the results
    today = date.today()
    if use_embed_layer:
        experiment = ' with aux net 3000 epochs'
    else:
        experiment = ' without aux net 3000 epochs'

    res_file_name = str(today) + experiment
    with open(res_file_name + '.pkl', 'wb') as f:
        pl.dump(results, f)
    print('Saved results data file: ', res_file_name + '.pkl')


def run_plot_results():
    batch_sizes = [64]

    file_name = 'results pkl/tests_res 800 epoch.pkl'
    experiment = 'losses without aux net 64'

    with open(file_name, 'rb') as f:
        results = pl.load(f)

    train_accs = list(r[2] for r in results)
    valid_accs = list(r[3] for r in results)
    test_accs = list(r[4] for r in results)

    train_losses = list(r[0] for r in results)
    valid_losses = list(r[1] for r in results)
    epoch_times = list(r[5] for r in results)
    train_time = list(r[6] for r in results)

    pr.plot_results(experiment, [train_losses[0], valid_losses[0]], batch_sizes, test_accs)


if __name__ == '__main__':
    test_train()
