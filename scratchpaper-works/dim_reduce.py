from emnist import extract_training_samples
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.decomposition import PCA
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


def SplitbyClasses(classSize=100, classes=None):
    """
        Given the labels you want and the maximal size for each of the label, this function
        will return the splited data set for each.
    :param classes:
        An array for the labels you want to filter out from the EMNISt data set.
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


def SymbolesToLabels(symbol:str, SymToLabel=dict()):
    assert len(symbol) == 1
    if len(SymToLabel) != 0:
        return SymToLabel[symbol]
    Letters = "".join([chr(97 + II) for II in range(26)])
    Digits = "".join(map(str, range(10)))
    for II, V in enumerate(Digits + Letters + Letters.upper()):
        SymToLabel[V] = II
    return SymToLabel[symbol]



class LDADimReduce:

    def __init__(this, X = None, y= None, n_components=60, classSize=1000, classes=None):
        if (X is None) or (y is None):
            X, y = SplitbyClasses(classSize=classSize, classes=classes)
        Template = LDA(n_components=n_components)
        lda = Template.fit(X, y)
        this.LdaModel = lda
        this.Dim = n_components
        this.ClassSize = classSize
        this.Data = X
        this.Labels = y

    def getEmbeddings(this, toTransform):
        lda = this.LdaModel
        Embeddings = lda.transform(toTransform)  # Rows are embeddings in all 60 dimensions
        return Embeddings

class PCADimReduce:

    def __init__(this,X=None, y=None, n_components=0.9, classSize=1000, classes=None):
        if (X is None) or (Y is None):
            X, y= SplitbyClasses(classSize=classSize, classes=classes)
        this.PcaModel = PCA(n_components=n_components, svd_solver="full")
        this.PcaModel.fit(X, y)
        this.classSize = 1000
        this.classes = classes
        this.Data = X
        this.Labels = y

    def getEmbeddings(this, X=None):
        if X is None:
            return this.PcaModel.transform(this.Data)
        else:
            return this.PcaModel.transform(X)

    def getCompEV(this):
        return this.PcaModel.explained_variance_ratio_



def main():
    def LDADemonstration():
        LdaInstance = LDADimReduce(); print(f"Geting the LDA Model... ")
        Data, Labels = SplitbyClasses(classSize=1000, classes=[26, 27, 28])
        # new data that never seemed before.
        Embeddings = LdaInstance.getEmbeddings(Data); print("Getting Embeddings...")
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
        title("LDA Separations on Test Set")
        show()
        print("Demonstration ended. ")

    LDADemonstration()

    def PCAPlusLDADemonstration():
        PCAInstance = PCADimReduce(n_components=0.8); print("Getting PCA Model...")
        PCAEmbeddings = PCAInstance.getEmbeddings(); print("Getting PCA Embeddings...")
        TrainLabels = PCAInstance.Labels
        print("Train LDA on PCA Embeddings... ")
        LDAMaxComponents = min(61, PCAInstance.PcaModel.n_components_ - 1)
        LDAInstance = LDADimReduce(n_components=LDAMaxComponents,  # Just to be sure.
                                   X=PCAEmbeddings,
                                   y=TrainLabels)  # Use the PCA embeddings to train LDA

        classes = [SymbolesToLabels(Char) for Char in "0o2z"]
        TestData, TestLabels = SplitbyClasses(classSize=1000, classes=classes)
        # Ge the LDA embeddings of the PCA embeddings.
        print("Represent using PCA modes and then on LDA basis... ")
        Embeddings = LDAInstance.getEmbeddings(PCAInstance.getEmbeddings(TestData))
        colors = ['navy', 'turquoise', 'darkorange', "purple"]
        SeparatingModes = [0, 1]
        for color, II in zip(colors, classes):
            scatter(
                Embeddings[TestLabels == II, SeparatingModes[0]],
                Embeddings[TestLabels == II, SeparatingModes[1]],
                alpha=.8,
                color=color
            )
        legend(list("0o2z"))
        title("PCA + LDA on Test data set")
        show()

    PCAPlusLDADemonstration()



if __name__ == "__main__":
    if CURRENT_DIRECTORY is not None:
        os.chdir(CURRENT_DIRECTORY)
    print(f"Script CWD: {os.getcwd()}")
    main()