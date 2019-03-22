import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def loadDataSet(fileName, delim = '\t'):
    fr = open(fileName)
    dataArr = []
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    for line in stringArr:
        dataArr.append(list(map(float, line)))
    return np.mat(dataArr)


def pca(dataMat, topNfeat = 9999999):
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = np.cov(meanRemoved, rowvar=0)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1): -1]
    redEigVects = eigVects[:, eigValInd]
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


if __name__ == "__main__":
    dataMat = loadDataSet('ch13_pca_testSet.txt')
    lowDMat, reconMat = pca(dataMat, 1)
    print(np.shape(lowDMat))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0],
               marker='^', s=90)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0],
               marker='o', s=50, c='red')
    plt.grid()
    plt.show()


