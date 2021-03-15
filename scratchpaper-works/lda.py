from emnist import extract_training_samples
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

CURRENT_DIRECTORY = None

## name space stuff:
shuffle = np.random.shuffle
reshape = np.reshape
zeros = np.zeros
scatter = plt.scatter
show = plt.show
concatenate = np.concatenate
mean = np.mean
figure = plt.figure
title = plt.title
legend = plt.legend

class LDADimReduce:

    def __init__(this, dim=60, classSize=1000, classes=None):
        X, y = PrepareDataForLDA(classSize=classSize, classes=classes)
        Template = LDA(n_components=dim)
        lda = Template.fit(X, y)
        this.LdaModel = lda
        this.Dim = dim
        this.ClassSize = classSize

    def getEmbeddings(this, toTransform):
        lda = this.LdaModel
        print("fitting lda")
        Embeddings = lda.transform(toTransform)  # Rows are embeddings in all 60 dimensions
        return Embeddings


def main():
    LdaInstance = LDADimReduce()

    Data, Labels = PrepareDataForLDA(classSize=1000, classes=[26, 27, 28])
    # new data that never seemed before.
    Embeddings = LdaInstance.getEmbeddings(Data)
    colors = ['navy', 'turquoise', 'darkorange']
    SeparatingModes = [0, 1]
    for color, II in zip(colors, [26, 27, 28]):
        scatter(
            Embeddings[Labels == II, SeparatingModes[0]],
            Embeddings[Labels == II, SeparatingModes[1]],
            alpha=.8,
            color=color
        )
    legend(["a", "b", "c"])
    show()


def PrepareDataForLDA(classSize=100, classes=None):
    """

    :param classSize:
        For each of the labels, what is the max size we want for each of the labels.
    """
    images, labels = extract_training_samples("byclass")
    Idx = []
    classes = list(range(0, 62)) if classes is  None else classes
    for Label in classes:
        ClsIdx = np.argwhere(labels == Label)
        ClsIdx = reshape(ClsIdx, ClsIdx.shape[0])
        ClsIdx = ClsIdx[: min(ClsIdx.shape[0], classSize)]
        for Idex in ClsIdx:
            Idx.append(Idex)
    RowDataMtx = images[Idx, :, :]
    Labels = labels[Idx]
    RowDataMtx = reshape(RowDataMtx, (len(Idx), 28*28))
    RowDataMtx = RowDataMtx.astype(np.float)
    RowDataMtx /= 255
    RowDataMtx -= mean(RowDataMtx, axis=1, keepdims=True)
    return RowDataMtx, Labels



if __name__ == "__main__":
    if CURRENT_DIRECTORY is not None:
        os.chdir(CURRENT_DIRECTORY)
    print(f"Script CWD: {os.getcwd()}")
    main()