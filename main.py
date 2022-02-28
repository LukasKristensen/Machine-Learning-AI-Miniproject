import tensorflow as tf
from tensorflow import keras
import numpy as np


def formatData(datasetPath):
    with open(datasetPath) as document:
        dataset = document.read().splitlines()
        return dataset


def main(dataset):
    if dataset is None:
        print("Did not register a dataset")
        return
    print(f"Evaluating dataset with the length {len(dataset)}")


if __name__ == '__main__':
    nasdaq71_22 = 'data-set/^IXIC 1971-2022.csv'
    datasetFormatted = formatData(nasdaq71_22)
    main(datasetFormatted)
