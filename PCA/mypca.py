import numpy as np
import pandas as pd

def getkthLargest(k, arr):
    if k == 1:
        return max(arr)
    m = max(arr)
    new_arr = list(filter(lambda a: a != m, arr))
    return getkthLargest(k - 1, new_arr)

def PCA_Self(X):
    X = np.transpose(X)
    covarianceMatrix = np.cov(X)

    curFeatures = 13
    wantedFeatures = 2

    eigenValues, eigenVectors = np.linalg.eig(covarianceMatrix)

    kthLargeEigen = getkthLargest(wantedFeatures, eigenValues)

    reducedPhi = np.zeros((wantedFeatures, curFeatures))
    count = 0

    for i in range(0, curFeatures):
        if (eigenValues[i] >= kthLargeEigen):
            reducedPhi[count, :] = eigenVectors[:, i]
            count += 1

    reducedDim = np.matmul(reducedPhi, X)
    reducedDim = np.transpose(reducedDim)
    return reducedDim



# Importing the dataset
dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:, 0:13].values
y=dataset.iloc[:,13].values

new_X = PCA_Self(X)
new_cov=np.cov(np.transpose(new_X))

eigenValues, eigenVectors = np.linalg.eig(np.cov(np.transpose(X)))