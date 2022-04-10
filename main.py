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
    print(dataset)

def kmeansclustering(stocksCSV):
    """
    The clustering code structure is based on the code from:
    https://pythonforfinance.net/2018/02/08/stock-clusters-using-k-means-algorithm-in-python/
    """

    ticker = stocksCSV['Ticker']
    Y5AVG = stocksCSV['5YAvgReturn']
    BETA = stocksCSV['Beta']
    PE = stocksCSV['PE']

    stockDataFrame = pd.DataFrame(Y5AVG)
    stockDataFrame.columns = ['Y5AVG']
    stockDataFrame['BETA'] = BETA
    stockDataFrame['PE'] = PE

    data = np.asarray([np.asarray(stockDataFrame['Y5AVG']), np.asarray(stockDataFrame['BETA']),np.asarray(stockDataFrame['PE'])]).T
    dataCopy = data

    print(stocksCSV.head())

    frameInertia = []
    for k in range(2, 6):
        kMeans = KMeans(n_clusters=k)
        kMeans.fit(dataCopy)
        frameInertia.append(kMeans.inertia_)

    plt.plot(range(2,6), frameInertia)
    plt.grid(True)
    plt.show()

    stockCentroids,_ = kmeans(data, 3)
    assignedStock,_ = vq(data, stockCentroids)

    fig3d = plt.figure(figsize=(15,15))
    axes = fig3d.add_subplot(111, projection='3d')
    axes.scatter(data[assignedStock==0,0], data[assignedStock==0,1],data[assignedStock==0,2], s=20, c='blue',label="Cluster 0")
    axes.scatter(data[assignedStock==1,0], data[assignedStock==1,1],data[assignedStock==1,2], s=20, c='green',label="Cluster 1")
    axes.scatter(data[assignedStock==2,0], data[assignedStock==2,1],data[assignedStock==2,2], s=20, c='red',label="Cluster 2")
    axes.scatter(data[assignedStock==3,0], data[assignedStock==3,1],data[assignedStock==3,2], s=20, c='yellow',label="Cluster 3")
    axes.set_xlabel('5YAvgReturn')
    axes.set_ylabel('Beta')
    axes.set_zlabel('PE')
    plt.show()

    for i in range(len(assignedStock)):
        print(assignedStock[i], ticker[i])

    """kmeanPlot.show()
    plt.show()"""
    print("View:","\n", stockDataFrame,assignedStock)


if __name__ == '__main__':
    nasdaq71_22 = 'data-set/^IXIC 1971-2022.csv'
    datasetFormatted = formatData(nasdaq71_22)

    filepathNasdaq = (open('data-set/^IXIC 1971-2022.csv'))
    nasdaqCSV = pd.read_csv(filepathNasdaq)

    filepathStocks = (open('data-set/testDataV3.csv'))
    stocksCSV = pd.read_csv(filepathStocks, sep=';')

    regression(nasdaqCSV)

    # kmeansclustering('data-set/testingData.csv')
    kmeansclustering(stocksCSV)
