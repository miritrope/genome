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


batch_sizes = [64]


def test_train(description):
    n_epochs = 1000
    patience = 50
    fold = 1
    use_embed_layer = False
    n_experiments = 1

    print('Training ... ')
    # blockPrint()

    results = []
    for i in range(n_experiments):
        for task in batch_sizes: results.append(lm.execute(fold, task, n_epochs, patience, use_embed_layer))

    enablePrint()
    for r in results: print(r, '\n')

    # save the results
    if use_embed_layer:
        experiment = 'with aux net '
    else:
        experiment = 'without aux net '

    res_file_name = experiment + description
    with open(res_file_name + '.pkl', 'wb') as f:
        pl.dump(results, f)
    print('Saved results data file: ', res_file_name + '.pkl')


def run_plot_results(file_name, experiment):

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
    experiment = 'without aux net ep 100 hi1 50 hi2 50 dr 08'
    test_train(experiment)
    run_plot_results(experiment + '.pkl', experiment)
