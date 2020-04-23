import learn_model as lm
import sys
import os
import pickle as pl
import plot_results as pr
import argparse


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

    res_file_name = exp_name + '_test_acc_' + f'{test_acc:.1f}'

    # save the results
    with open(res_file_name + '.pkl', 'wb') as f:
        pl.dump(results, f)
    print('Saved results data file: ', res_file_name + '.pkl')

    pic_name = res_file_name + '.png'
    pr.plot_results([train_losses, valid_losses],
                    n_epochs, pic_name)


def main():

    parser = argparse.ArgumentParser(description="""Train Genetic Networks""")

    parser.add_argument('-patience',
                        type=int,
                        default=50,
                        help="patience of the early stopping")
    parser.add_argument('-fold',
                        type=int,
                        default=1,
                        help="which fold of the dataset")
    parser.add_argument('-use_embed_layer',
                        type=int,
                        default=0,
                        help="whether to use the auxiliary network")
    parser.add_argument('-n_epochs',
                        type=int,
                        default=1000,
                        help="number of epochs")
    parser.add_argument('-batch_size',
                        type=int,
                        default=128,
                        help="batch size [32, 64, 128, 256]")
    parser.add_argument('-hidden_sizes',
                        type=int,
                        default=50,
                        help="hidden units sizes")
    parser.add_argument('-dropout_1',
                        type=float,
                        default=0.8,
                        help="dropout hidden layer 1")
    parser.add_argument('-dropout_2',
                        type=float,
                        default=0.5,
                        help="dropout hidden layer 2")
    parser.add_argument('-file_name',
                        type=str,
                        default='affy_6_biallelic_snps_maf005_thinned_aut_dataset.pkl',
                        help="file name of the genome and labels")
    args = parser.parse_args()

    # define experiment name
    experiment = ''
    if args.use_embed_layer:
        experiment += 'with_'
    else:
        experiment += 'without_'

    n_hidden = [args.hidden_sizes, args.hidden_sizes]
    drop_sizes = [args.dropout_1, args.dropout_2]

    experiment += ('ep_' + str(args.n_epochs) +
                    '_bs_' + str(args.batch_size) +
                        '_hi_' + str(n_hidden) +
                            '_dr_' + str(drop_sizes))

    print(experiment)

    results = lm.execute(args.file_name, args.batch_size, args.fold, n_hidden, args.n_epochs,
               args.patience, args.use_embed_layer, drop_sizes)

    run_results(results, experiment, args.n_epochs)


if __name__ == '__main__':
    main()
