import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import pyplot as plt

import pandas as pd
from sklearn.cluster import KMeans

def formatData(datasetPath):
    with open(datasetPath) as document:
        dataset = document.read().splitlines()
        return dataset


def regression(dataset):
    if dataset is None:
        print("Did not register a dataset")
        return
    print(f"Evaluating dataset with the length {len(dataset)}")
    print(dataset[0])
    print(dataset[1])
    print(dataset[-1])

def kmeansclustering(datasetPath):
    print("Clustering Dataset:",datasetPath)

    pureStock = {'TICK':['MSFT','AAPL','BABA','NVDA'],
                 '5YAVG':[85.71,89.02,0.03,128.06],
                 'BETA':[0.91,1.19,0.89,1.42]}

    stockData = pd.read_csv(datasetPath,index_col=0)
    stockDataFrame = pd.DataFrame(pureStock)
    print(stockDataFrame.head())


    frameInertia = []
    for k in range(2, 4):
        kMeans = KMeans(n_clusters=k)
        kMeans.fit(stockDataFrame)
        frameInertia.append(kMeans.inertia_)

    figureInertia = plt.figure(figusize=(10,5))
    plt.plot(range(2,4), frameInertia)
    plt.grid(True)
    plt.show()




if __name__ == '__main__':
    nasdaq71_22 = 'data-set/^IXIC 1971-2022.csv'
    datasetFormatted = formatData(nasdaq71_22)
    regression(datasetFormatted)

    kmeansclustering('data-set/testingData.csv')
