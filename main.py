import tensorflow as tf
from tensorflow import keras
import sklearn
import numpy as np

# Approaches: Deep Learning, Broker Fees, Discrete Fourier Transformation


def formatData(datasetPath):
    with open(datasetPath) as document:
        dataset = document.read().splitlines()
        return dataset


def main(dataset):
    if dataset is None:
        print("Did not register a dataset")
        return
    print(f"Evaluating dataset with the length {len(dataset)}")
    print(dataset[0])
    print(dataset[1])
    print(dataset[len(dataset)-1])


if __name__ == '__main__':
    nasdaq71_22 = 'data-set/^IXIC 1971-2022.csv'
    datasetFormatted = formatData(nasdaq71_22)
    main(datasetFormatted)
