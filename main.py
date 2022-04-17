from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from scipy.cluster.vq import kmeans,vq

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score

# Todo: Comment code before hand-in


def kmeansclustering(stocksCSV):
    """
    The clustering code structure is based on the code from:
    https://pythonforfinance.net/2018/02/08/stock-clusters-using-k-means-algorithm-in-python/
    """

    # Todo: Normalize features to fit -1 to 1
    ticker = stocksCSV['Ticker']
    Y5AVG = stocksCSV['5YAvgReturn']
    BETA = stocksCSV['Beta']
    PE = stocksCSV['PE']

    stockDataFrame = pd.DataFrame()
    stockDataFrame['Y5AVG'] = Y5AVG
    stockDataFrame['BETA'] = BETA
    stockDataFrame['PE'] = PE
    stockDataFrame['TICKER'] = ticker

    data = np.asarray([np.asarray(stockDataFrame['Y5AVG']), np.asarray(stockDataFrame['BETA']),np.asarray(stockDataFrame['PE'])]).T

    numberOfClusters = 4
    frameInertia = []

    for k in range(1, numberOfClusters+1):
        stockCentroids, inertia = kmeans(data, k)
        frameInertia.append(inertia)
        print(f'Euclidean distance with {k} clusters:',inertia)

    plt.plot(range(1,numberOfClusters+1), frameInertia)
    plt.grid(True)
    plt.show()

    stockCentroids,euclidean = kmeans(data, numberOfClusters)
    print("Euclidean distance:",euclidean)
    assignedStock, innerEuclidean = vq(data, stockCentroids)
    print("For-each data-point - Euclidean distance to cluster:",innerEuclidean)

    for i in range(len(assignedStock)):
        print(assignedStock[i], ticker[i])

    print("View:", "\n", stockDataFrame, assignedStock)

    stockDataFrame['CLUSTER'] = assignedStock
    print('Clustered:\n',stockDataFrame)

    fig3d = plt.figure(figsize=(9,7))
    axes = fig3d.add_subplot(111, projection='3d')
    axes.scatter(data[assignedStock==0,0], data[assignedStock==0,1],data[assignedStock==0,2], s=20, c='blue',label="Cluster 0")
    axes.scatter(data[assignedStock==1,0], data[assignedStock==1,1],data[assignedStock==1,2], s=20, c='green',label="Cluster 1")
    axes.scatter(data[assignedStock==2,0], data[assignedStock==2,1],data[assignedStock==2,2], s=20, c='red',label="Cluster 2")
    axes.scatter(data[assignedStock==3,0], data[assignedStock==3,1],data[assignedStock==3,2], s=20, c='purple',label="Cluster 3")
    axes.scatter(stockCentroids[:,0], stockCentroids[:,1], stockCentroids[:,2], s=80, c='black', alpha=0.3,label= "Centroids")
    axes.set_xlabel('AvgAnnualReturn(5Y)')
    axes.set_ylabel('Beta')
    axes.set_zlabel('PE')

    plt.show()


def regression(dataset):
    """
    Interesting ways to implementing regression:
    - Support Vector Machine (Should be the first on to try out)
    - Convolutional Neural Network
    - Recurrent Neural Network

    Material for implementing:
    - https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/
    - https://medium.com/pursuitnotes/support-vector-regression-in-6-steps-with-python-c4569acd062d
    - https://www.geeksforgeeks.org/predicting-stock-price-direction-using-support-vector-machines/?ref=rp
    - https://medium.com/@rupesh1684/stock-market-prediction-using-machine-learning-model-svm-e4aaca529886
    """

    if dataset is None:
        print("Did not register a dataset")
        return

    nasdaqDF = pd.DataFrame()
    nasdaqDF['Open'] = dataset['Open'].astype(int)
    nasdaqDF['Date'] = dataset['Date']
    nasdaqDF['Prediction'] = dataset['Open'].astype(int).shift(-10)
    nasdaqDF = nasdaqDF[:-10]


    print(f"Evaluating dataset with the length {len(dataset)}")
    print(dataset)
    print(nasdaqDF.head())
    plt.plot(dataset['Date'], dataset['Open'])
    plt.xticks(range(0,len(dataset['Date']),3000))
    plt.title("Nasdaq Composite 1971-2022")
    plt.show()

    trainingData = nasdaqDF[-400:-70]
    testData = nasdaqDF[-70:-40]

    xreshaped = np.array(trainingData['Open']).reshape(-1, 1)
    yreshaped = np.array(trainingData['Prediction'])

    xtestData = np.array(testData['Open']).reshape(-1, 1)
    ytestData = np.array(testData['Prediction'])

    # plt.plot(xreshaped, yreshaped)

    # print("x_Reshaped:",xreshaped)
    # print("y_Reshaped:",yreshaped)
    print("Debugging ...")

    # Check parameters: C and gamma
    svrClass = SVR(kernel='rbf').fit(xreshaped, yreshaped)
    testPredict = svrClass.predict([[13200]])
    print("Done SVC",testPredict)
    print("Score: ",svrClass.score(xtestData,ytestData))
    plt.show()


if __name__ == '__main__':
    filepathNasdaq = (open('data-set/^IXIC 1971-2022.csv'))
    nasdaqCSV = pd.read_csv(filepathNasdaq)
    regression(nasdaqCSV)

    filepathStocks = (open('data-set/testDataV3.csv'))
    stocksToCSV = pd.read_csv(filepathStocks, sep=';')
    kmeansclustering(stocksToCSV)
