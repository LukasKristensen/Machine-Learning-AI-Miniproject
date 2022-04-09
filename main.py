import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.vq import kmeans,vq

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
    """
    The clustering structure is based on the code from:
    https://pythonforfinance.net/2018/02/08/stock-clusters-using-k-means-algorithm-in-python/
    """

    ticker = ['MSFT', 'AAPL', 'BABA', 'NVDA', 'PINS', 'ZM', 'SHOP', 'PYPL', 'MA', 'ADBE', 'TSLA']
    Y5AVG = [85.71,89.02,0.03,128.06,0.01,0.08,31.32,43.6,57.33,62.79]
    BETA = [0.91,1.19,0.89,1.42,1.19,-0.94,1.62,1.29,1.08,1.05]
    PE = [31.61,28.28,27.54,60.05,51.2,24.97,26.33,31.59,40.21,44.23]

    print(len(Y5AVG),len(BETA),len(PE))

    stockDataFrame = pd.DataFrame(Y5AVG)
    stockDataFrame.columns = ['Y5AVG']
    stockDataFrame['BETA'] = BETA
    stockDataFrame['PE'] = PE

    data = np.asarray([np.asarray(stockDataFrame['Y5AVG']), np.asarray(stockDataFrame['BETA']), np.asarray(stockDataFrame['PE'])]).T
    dataCopy = data

    print(stockDataFrame.head())

    frameInertia = []
    for k in range(2, 6):
        kMeans = KMeans(n_clusters=k)
        kMeans.fit(dataCopy)
        frameInertia.append(kMeans.inertia_)

    plt.plot(range(2,6), frameInertia)
    plt.grid(True)
    plt.show()

    stockCentroids,_ = kmeans(data, 4)
    assignedStock,_ = vq(data, stockCentroids)

    kmeanPlot = plt
    kmeanPlot.plot(data[assignedStock==0,0], data[assignedStock==0,1], 'ob',
                   data[assignedStock==1,0], data[assignedStock==1,1], 'oy',
                   data[assignedStock==2,0], data[assignedStock==2,1], 'og',
                   data[assignedStock==3,0], data[assignedStock==3,1], 'or')

    kmeanPlot.plot(stockCentroids[:,0], stockCentroids[:,1],'sg',markersize=10, alpha=0.1)

    for i in range(len(assignedStock)):
        print(assignedStock[i], ticker[i])

    kmeanPlot.show()
    plt.show()
    print("View:","\n", stockDataFrame,assignedStock)



if __name__ == '__main__':
    nasdaq71_22 = 'data-set/^IXIC 1971-2022.csv'
    datasetFormatted = formatData(nasdaq71_22)
    regression(datasetFormatted)

    kmeansclustering('data-set/testingData.csv')
