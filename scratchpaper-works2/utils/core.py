from emnist import extract_training_samples, extract_test_samples
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

## name space stuff
diag = np.diag
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
matshow = plt.matshow
array_equal=np.array_equal
unique = np.unique
ylim = plt.ylim

## Meta settings
CURRENT_DIRECTORY = None
UPPER_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOWER_LETTERS = "abcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"
matplotlib.rcParams['figure.figsize'] = (10, 10)
matplotlib.rcParams["figure.dpi"] = 220


def SplitbyClasses(classSize=100, classes=None, shuffle_data=False, test_set=False):
    """
        Given the labels you want and the maximal size for each of the label, this function
        will return the splited data set for each.
        Note:
            Data is chosen from the training set.
    :param classes:
        An array for the labels you want to filter out from the EMNISt data set.
    :param classSize:
        For each of the labels, what is the max size we want for each of the labels.
    """
    if test_set:
        images, labels = extract_test_samples("byclass")
    else:
        images, labels = extract_training_samples("byclass")
    Idx = []
    classes = list(range(0, 62)) if classes is  None else classes
    for Label in classes:
        ClsIdx = np.argwhere(labels == Label)
        ClsIdx = reshape(ClsIdx, ClsIdx.shape[0])
        if shuffle_data: shuffle(ClsIdx)
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


def SplitbyLetterDigits(classSize=100, test_set=False):
    """
    Split the training/test data set into 3 big groups:
        1. The digits.
        2. The Upper cased letters.
        3. The lower cased letters.

    Note: This class will always shuffle the data!
    Note: This class will perform zero mean unit variances STANDARDIZATION.

    :param classSize:
        The maximal size for each of the classes.
    :param test_set:
        Whether to choose the data set from the EMNISt test set, or the data set.
    :Return :
        The returned labels will still be in the range of [0, 62], but they
        will be evenly partitioned into the 3 classes above.
    """
    if test_set:
        images, labels = extract_test_samples("byclass")
    else:
        images, labels = extract_training_samples("byclass")

    Organizer = LabelsOrganizer()
    TreeClassesLabels = Organizer.getDataLabels(labels)
    IndicesChosen = []
    for II in range(3):
        Idx = np.where(TreeClassesLabels==II)
        Idx = Idx[0]
        shuffle(Idx)
        Idx = Idx[:min(len(Idx), classSize)]
        for JJ in Idx:
            IndicesChosen.append(JJ)

    RowDataMtx = images[IndicesChosen, ::, ::]
    Labels = labels[IndicesChosen]
    RowDataMtx = reshape(RowDataMtx, (len(IndicesChosen), 28 * 28))
    RowDataMtx = RowDataMtx.astype(np.float)
    RowDataMtx /= 255
    RowDataMtx -= mean(RowDataMtx, axis=1, keepdims=True)
    return RowDataMtx, Labels


def SymbolsToLabels(symbol:str, SymToLabel=dict()):
    assert len(symbol) == 1
    if len(SymToLabel) != 0:
        return SymToLabel[symbol]
    Letters = "".join([chr(97 + II) for II in range(26)])
    Digits = "".join(map(str, range(10)))
    for II, V in enumerate(Digits + Letters.upper() + Letters):
        SymToLabel[V] = II
    return SymToLabel[symbol]


def LabelsToSymbols(label:int, LabelToSymb=dict()):
    if len(LabelToSymb) != 0:
        return LabelToSymb[label]
    Letters = "".join([chr(97 + II) for II in range(26)])
    Digits = "".join(map(str, range(10)))
    for II, V in enumerate(Digits + Letters.upper() + Letters):
        LabelToSymb[II] = V
    return LabelToSymb[label]


class ConfusionMatrix:

    def __init__(this, testLabels, predictedlabels, axisTicks=None):
        this.TestLabels = testLabels
        this.PredictedLabels = predictedlabels
        Conmat = confusion_matrix(testLabels, predictedlabels)
        this.ConfusionMatrix = Conmat
        this.TotalAccuracy = \
            np.sum(diag(Conmat))/np.sum(Conmat)
        # False positive, none diag row sum, it is but actually it's not.
        this.FalsePositiveEach = np.sum(Conmat - diag(diag(Conmat)), axis=0)/(np.sum(Conmat, axis=0))
        # False negative, none diag column sum, it's not but actually it is.
        this.FalseNegativeEach = np.sum(Conmat - diag(diag(Conmat)), axis=1)/(np.sum(Conmat, axis=1))
        LabelsSorted = unique(this.TestLabels.copy())
        np.sort(LabelsSorted)
        if axisTicks is None:
            Ticks = [LabelsToSymbols(II) for II in LabelsSorted]
            this.AxisTicks = Ticks
        else:
            this.AxisTicks = axisTicks

    def visualize(this, title:str=None):
        """
            Visualize the confusion matrix.
        :return:
            fig, ax
        """
        fig = figure()
        ax = fig.add_subplot(111)
        im = ax.matshow(this.ConfusionMatrix, cmap='winter', interpolation='nearest')
        fig.colorbar(im)

        for (i, j), z in np.ndenumerate(this.ConfusionMatrix):
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', color="white",
                bbox=dict(facecolor='none', edgecolor='0.3'))

        plt.xlabel("Predicted")
        plt.ylabel("True Labels")
        ax.set_xticks(np.arange(len(this.AxisTicks)))
        ax.set_xticklabels(this.AxisTicks)
        ax.set_yticks(np.arange(len(this.AxisTicks)))
        ax.set_yticklabels(this.AxisTicks)
        if title is not None:
            fig.suptitle(title)
        return fig, ax

    def report(this, sort_it=True):

        FPAxTicks = this.AxisTicks
        FNAxTicks = this.AxisTicks
        FP = this.FalsePositiveEach
        FN = this.FalseNegativeEach

        if sort_it:
            ToSort = list(zip(FP, FPAxTicks))
            ToSort = sorted(ToSort, key=lambda x: x[0])
            FP, FPAxTicks = zip(*ToSort)

            ToSort = list(zip(FN, FNAxTicks))
            ToSort = sorted(ToSort, key=lambda x: x[0])
            FN, FNAxTicks = zip(*ToSort)

        print(f"Over all accuracy is: {this.TotalAccuracy}")
        fig, (Top, Bottom) = plt.subplots(2, 1)
        fig.suptitle("Accuracy Each Label")
        Top.bar(FPAxTicks, FP)
        Top.set_title("False Positive")
        Top.set_ylim((0, 1))
        Bottom.bar(FNAxTicks, FN)
        Bottom.set_title("False Negative")
        Bottom.set_ylim((0, 1))
        return fig, (Top, Bottom)

class LabelsOrganizer:
    """
    Group all the labels by:
        1. upper cased letters,
        2. lower cased letters,
        1. digits.
    """

    def __init__(this, mapping:callable=None, axisTicks:dict=None):
        def DefaultMapping(label):
            if LabelsToSymbols(label) in DIGITS:
                return 0
            elif LabelsToSymbols(label) in LOWER_LETTERS:
                return 1
            elif LabelsToSymbols(label) in UPPER_LETTERS:
                return 2
            else:
                raise Exception("Something is wrong this should not happen check code. ")

        if (mapping is None) != (axisTicks is None):
            raise Exception("The new map for labels and the label mapping function has to be given at the same time.")

        if mapping is None:
            this.Mapping = DefaultMapping
        else:
            this.Mapping = mapping

        if axisTicks is None:
            this.AxisTicks = {0: "Digits", 1: "Lower Cased Letter", 2: "Upper Letters"}
        else:
            this.AxisTicks = axisTicks

    def getDataLabels(this, labels):
        Labels = np.vectorize(this.Mapping)(labels)
        return Labels

    def getDataTicks(this):
        """
        Returns: the axis labels for the confusion matrix.
        """

        return list(this.AxisTicks.values())


class LDADimReduce:

    def __init__(this,
                 X = None,
                 y= None,
                 n_components=None,
                 classSize=1000,
                 classes=None,
                 shuffle=False
                 ):
        """
            Creates an instance of the LDA dim_reduce on the EMNIST data set.
        :param X: (Optional) The data that we want to train the LDA on.
        :param y: (Optional, require X) The labels of the training data X.
        :param n_components:
            The number of components we want to for the embeddings.
        :param classSize:
            The size of the class to sample. All classes will be sampled by the same amount.
        :param classes:
            List of labels we want to sample from the EMNIST data set.
        """
        if (X is None) or (y is None):
            X, y = SplitbyClasses(classSize=classSize, classes=classes, shuffle_data=shuffle)
        Template = LDA(n_components=n_components)
        lda = Template.fit(X, y)
        this.LdaModel = lda
        this.Dim = n_components
        this.ClassSize = classSize
        this.Data = X
        this.Labels = y

    def getEmbeddings(this, toTransform=None):
        """
        Get the embeddings for the given data
        :param toTransform: (Optional) if not given, it will return the embeddings of the data that the
        LDA model is trained on.
        :return:
            The embeddings of the data. Each row is a sample.
        """
        lda = this.LdaModel
        if toTransform is not None:
            Embeddings = lda.transform(toTransform)  # Rows are embeddings in all 60 dimensions
            return Embeddings
        else:
            return lda.transform(this.Data)


class PCADimReduce:

    def __init__(this,
                 X=None,
                 y=None,
                 n_components=0.9,
                 classSize=1000,
                 classes=None,
                 shuffle=False
                 ):
        """
            Create an instance for the PCA dimn reduce embedding.
        :param X:
        :param y:
        :param n_components:
        :param classSize:
        :param classes:
        """
        if (X is None) or (y is None):
            X, y= SplitbyClasses(classSize=classSize, classes=classes, shuffle_data=shuffle)
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

class DimReduceHybrid:
    """
    PCA + LDA embeddings.
    """
    def __init__(this, X=None, y=None, classes=None, classSize=1000, pca_components=0.9, shuffle=False):
        if (X is None) or (y is None):
            X, y = SplitbyClasses(classSize=classSize, classes=classes, shuffle_data=False)

        this.PCAModel = PCA(n_components=pca_components)
        this.PCAModel.fit(X, y)
        PCAEmbeddings = this.PCAModel.transform(X)
        this.LDAModel = LDA()
        this.LDAModel.fit(PCAEmbeddings, y)

        this.classSize = 1000
        this.pca_components=0.9
        this.Data = X
        this.Labels = y

    def getEmbeddings(this, X=None):
        X = this.Data if X is None else X
        PCAEmbeddings = this.PCAModel.transform(X)
        LDAEmbeddings = this.LDAModel.transform(PCAEmbeddings)
        return LDAEmbeddings


def main():
    pass


if __name__ == "__main__":
    if CURRENT_DIRECTORY is not None:
        os.chdir(CURRENT_DIRECTORY)
    print(f"Script CWD: {os.getcwd()}")
    main()