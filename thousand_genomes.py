import pickle as pl


def load_data(path):
    with open(path, "rb") as f:
        genomic_data, label_data = pl.load(f)
        print('loaded the 1k genome raw data pickle file')

    return genomic_data, label_data


if __name__ == '__main__':
    x = load_data()
    print("Loading Done")
