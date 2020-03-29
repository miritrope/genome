import thousand_genomes
import numpy as np
import os
# main purpose: loading data and manipulate its values by shuffling and splitting


def shuffle(data_sources, seed=23):
    """
    Shuffles multiple data sources (numpy arrays) together so the
    correspondance between data sources (such as inputs and targets) is
    maintained.
    """
    np.random.seed(seed)
    indices = np.arange(data_sources[0].shape[0])
    np.random.shuffle(indices)

    return [d[indices] for d in data_sources]


def split(data_sources, splits):
    """
    Splits the given data sources (numpy arrays) according to the provided
    split boundries.

    Ex : if splits is [0.6], every data source will be separated in two parts,
    the first containing 60% of the data and the other containing the
    remaining 40%.
    """

    if splits is None:
        return data_sources

    split_data_sources = []
    n_samples = data_sources[0].shape[0]
    start = 0
    end = 0

    for s in splits:
        end += int(n_samples * s)
        split_data_sources.append([d[start:end] for d in data_sources])
        start = end
    split_data_sources.append([d[end:] for d in data_sources])

    return split_data_sources


def load_1000_genomes(data_path, raw_path, fold=0):
    x, y = thousand_genomes.load_data(data_path + raw_path)
    x = x.astype("float32")

    # Shuffle the 1k genome raw data
    (x, y) = shuffle((x, y))
    all_folds = split([x, y], [.2, .2, .2, .2])

    # Split the data into 5 classes of 20% each
    assert fold >= 0
    assert fold < 5

    # separate test and all other folds
    test = all_folds[fold]
    all_folds = all_folds[:fold] + all_folds[(fold + 1):]

    # concatenate all folds
    x = np.concatenate([el[0] for el in all_folds])
    y = np.concatenate([el[1] for el in all_folds])

    # Data used for supervised training
    train, valid = split([x, y], [0.75])
    # .75 train .25 valid

    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    train[0] = (train[0] - mu[None, :]) / sigma[None, :]
    valid[0] = (valid[0] - mu[None, :]) / sigma[None, :]
    test[0] = (test[0] - mu[None, :]) / sigma[None, :]

    # supervised vector
    sup = [train, valid, test]
    return sup


def load_embedding_mat(dataset_path, emb_path, fold, transpose):

    try:
        unsupervised_data = np.load(os.path.join(dataset_path, emb_path +
                                                    str(fold) + '.npy'))
        if transpose:
            unsupervised_data = unsupervised_data.transpose()

        feat_emb_val = unsupervised_data.astype('float32')

    except IOError:
        print("The embedding matrix file does not exist: ", os.path.join(dataset_path, emb_path +
                                                    str(fold) + '.npy'))

    return feat_emb_val


if __name__ == '__main__':
    print("Load data")
    x = load_1000_genomes(data_path='data/', raw_path='affy_6_biallelic_snps_maf005_thinned_aut_dataset.pkl', fold=0)
    emb = load_embedding_mat(dataset_path='data/', emb_path='histo3x26_fold', fold=0, transpose=True)
