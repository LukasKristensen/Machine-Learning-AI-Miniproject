import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.cluster.vq import kmeans, vq

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error

def kMeansClustering(stocksCSV):
    """
    The clustering code structure is based on the code from:
    https://pythonforfinance.net/2018/02/08/stock-clusters-using-k-means-algorithm-in-python/
    """

    # Todo: Normalize features to fit -1 to 1

    # Defining the features
    ticker = stocksCSV['Ticker']
    Y5AVG = stocksCSV['5YAvgReturn']
    BETA = stocksCSV['Beta']
    PE = stocksCSV['PE']

    # Loading the data into a pandas DataFrame
    stockDataFrame = pd.DataFrame()
    stockDataFrame['Y5AVG'] = Y5AVG
    stockDataFrame['BETA'] = BETA
    stockDataFrame['PE'] = PE
    stockDataFrame['TICKER'] = ticker

    # Loading the data into a numpy array
    data = np.asarray([np.asarray(stockDataFrame['Y5AVG']), np.asarray(stockDataFrame['BETA']), np.asarray(stockDataFrame['PE'])]).T

    # Defining the amount of clusters and preparing an array for the Inertia
    numberOfClusters = 3
    evaluateAmountEuclidean = []

    for k in range(1, numberOfClusters+1):
        stockCentroids, euclidean = kmeans(data, k)
        evaluateAmountEuclidean.append(euclidean)
        print(f'Euclidean distance with {k} clusters:', euclidean)

    # Plotting the iterative evaluation over the amount of clusters relative to inertia
    plt.plot(range(1, numberOfClusters+1), evaluateAmountEuclidean)
    plt.grid(True)
    plt.show()

    # Loading the euclidean again. This time only the final model
    stockCentroids, euclidean = kmeans(data, numberOfClusters)
    print("Euclidean distance:", euclidean)
    assignedStock, innerEuclidean = vq(data, stockCentroids)
    print("For-each data-point - Euclidean distance to cluster:", innerEuclidean)

    # Combining the assigned clusters to the rest of the dataframe and prints the result
    stockDataFrame['CLUSTER'] = assignedStock
    print('Clustered:\n', stockDataFrame)

    # Plotting the clusters into Matplotlib with the assigned stocks with color coding. Setting the projection to 3D
    fig3d = plt.figure(figsize=(9,7))
    axes = fig3d.add_subplot(111, projection='3d')
    axes.scatter(data[assignedStock==0,0], data[assignedStock==0,1],data[assignedStock==0,2], s=20, c='blue',label="Cluster 0")
    axes.scatter(data[assignedStock==1,0], data[assignedStock==1,1],data[assignedStock==1,2], s=20, c='green',label="Cluster 1")
    axes.scatter(data[assignedStock==2,0], data[assignedStock==2,1],data[assignedStock==2,2], s=20, c='red',label="Cluster 2")
    axes.scatter(data[assignedStock==3,0], data[assignedStock==3,1],data[assignedStock==3,2], s=20, c='purple',label="Cluster 3")
    axes.scatter(stockCentroids[:,0], stockCentroids[:,1], stockCentroids[:,2], s=80, c='black', alpha=0.3,label= "Centroids")

    # Defining the features relative to the axis on the plot
    axes.set_xlabel('AvgAnnualReturn(5Y)')
    axes.set_ylabel('Beta')
    axes.set_zlabel('PE')

    plt.show()


def regression(dataset):
    if dataset is None:
        print("Did not register a dataset")
        return

    # Data-set parameters
    daysPredict = 20

    # Loading the data-set into a pandas DataFrame
    nasdaqDF = pd.DataFrame()
    nasdaqDF['Open'] = dataset['Open'].astype(int)
    nasdaqDF['Date'] = dataset['Date']
    nasdaqDF['Prediction'] = dataset['Open'].shift(-daysPredict)
    nasdaqDF = nasdaqDF[:-daysPredict]

    # Viewing the data-set
    plt.plot(dataset['Date'], dataset['Open'])
    plt.xticks(range(0, len(dataset['Date']), 3000))
    plt.title("Nasdaq Composite 1971-2022")
    plt.show()

    # Splitting the data-set into training data and test data
    trainingData = nasdaqDF[:-2000]
    validationData = nasdaqDF[-4000:-2000]
    testData = nasdaqDF[-2000:]

    # Normalizing features
    xReshaped = np.array(trainingData['Open']).reshape(-1, 1)
    yReshaped = np.array(trainingData['Prediction']).reshape(-1, 1).flatten()
    xTestData = np.array(testData['Open']).reshape(-1, 1)
    yTestData = np.array(testData['Prediction']).reshape(-1, 1)

    # Setting up and fitting the model
    """
    Next 6 lines (SVR-Model) is based on the documentation from: 
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html"""
    modelRegression = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=2000, gamma=0.001))
    fittedRegression = modelRegression.fit(xReshaped, yReshaped)

    # Predicting the test-data
    predictionRegression = fittedRegression.predict(xTestData)

    # Evaluating the model
    scoreReg = explained_variance_score(yTestData.flatten(), predictionRegression)
    meanError = mean_absolute_error(yTestData.flatten(), predictionRegression)
    score = modelRegression.score(xTestData,yTestData)
    print("Variance Score:",scoreReg)
    print("Mean Error:",meanError)
    print("R2 Score:",score)

    # Visualizing the prediction compared to the true data, combined with historical trained data
    plt.plot(dataset['Date'][-1980:], predictionRegression[daysPredict:], c="red")
    plt.plot(yTestData[:-daysPredict], c="green")
    plt.xticks(range(0, 2000, 300))
    plt.show()


if __name__ == '__main__':
    # Regression Model
    filepathNasdaq = (open('data-set/^IXIC 1971-2022.csv'))
    nasdaqCSV = pd.read_csv(filepathNasdaq, sep=',')
    regression(nasdaqCSV)

    # Clustering Model
    filepathStocks = (open('data-set/testDataV3.csv'))
    stocksToCSV = pd.read_csv(filepathStocks, sep=';')
    kMeansClustering(stocksToCSV)
