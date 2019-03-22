import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
def loadExData():
    return [[1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1]]


def ecludSim(inA, inB):
    return 1.0/(1.0 + la.norm(inA - inB))


def pearsSim(inA, inB):
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=0)[0][1]


def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num/denom)


def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print(1, end=' ')
            else:
                print(0, end=' ')
        print(' ')


def imgCompress(numSv=3, thresh=0.8):
    myl = []
    myMat = []
    for line in open('ch14_5_65.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = np.mat(myl)
    # print('***original matrix***')
    # printMat(myMat, thresh)
    U, Sigma, VT = la.svd(myMat)
    SigRecon = np.mat(np.zeros((numSv, numSv)))
    for k in range(numSv):
        SigRecon[k, k] = Sigma[k]
    reconMat = U[:, :numSv] * SigRecon * VT[:numSv, :]
    # print('***reconstructed matrix using singular values******')
    # printMat(reconMat, thresh)
    fig = plt.figure()
    im1 = fig.add_subplot(121)
    im1.imshow(myMat)
    im1.set_title("origin matrix")
    im2 = fig.add_subplot(122)
    im2.imshow(reconMat)
    im2.set_title("reconstructed matrix")
    # plt.axis('off')  # 不显示坐标轴
    # pylab.show()
    plt.show()



if __name__ == '__main__':
    # Data = loadExData()
    # U, Sigma, VT = np.linalg.svd(Data)
    # print(Sigma)
    imgCompress(2)

