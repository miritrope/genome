import pickle as pl
import os
from os import path


def load_data(path):
    try:
        with open(path, "rb") as f:
            # Load the 1K genome raw data pickle file
            genomic_data, label_data = pl.load(f)
    except IOError:
        print("The 1k genome file does not exist: ",path)

    return genomic_data, label_data


if __name__ == '__main__':
    print("Load Data")
    x = load_data()
