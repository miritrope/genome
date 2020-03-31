import dataset_utils as du
import numpy as np
import os


def generate_embedding(data_path, raw_path, norm, fold):

    train, valid, test = du.load_1000_genomes(data_path, raw_path, norm, fold=fold)

    genome = np.vstack([train[0], valid[0]]).transpose()
    labels = np.vstack([train[1], valid[1]])

    n_feats = genome.shape[0]
    # from one-hot matrix to integers
    labels = labels.argmax(axis=1)

    filename = 'embed_4x26'
    filename += '_fold' + str(fold) + '.npy'

    n_genotypes = 4
    n_targets = 26

    embed = np.zeros((n_feats, n_genotypes * n_targets))
    for i in range(embed.shape[0]):
        if i % 5000 == 0:
            print("processing snp no: ", i)
        for j in range(n_targets):
            # generate for each snp , a histogram of four bins, one for each allel's option
            embed[i, j*n_genotypes: j*n_genotypes + n_genotypes] += \
                np.bincount(genome[i, labels == j].astype('int32'), minlength=n_genotypes)

            # normalizing the result for each class
            embed[i, j*n_genotypes: j*n_genotypes + n_genotypes] /= \
                embed[i, j*n_genotypes: j*n_genotypes + n_genotypes].sum()

        embed = embed.astype('float32')

    np.save(os.path.join(data_path, filename), embed)


# By default, make 5 folds to calculate the mean mis-classification error of the training sessions
if __name__ == '__main__':
    for f in range(5):
        print('Generate fold number: ', f)
        data_path = 'data/'
        raw_path = 'affy_6_biallelic_snps_maf005_thinned_aut_dataset.pkl'
        generate_embedding(data_path, raw_path, norm=False, fold=f)
        print('made 5 folds datasets')
