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

def main():
    X, y = PrepareData(classSize=1000)
    lda = LDA(n_components=60)
    print("fitting lda")
    Subspace = lda.fit(X, y).transform(X)  # Rows are embeddings in all 60 dimensions
    print(f"Shape of the subspace: {Subspace.shape}")
    print(Subspace)
    colors = ['navy', 'turquoise', 'darkorange']
    for color, II in zip(colors, [0, 1, 9]):
        scatter(Subspace[y == II, 0], Subspace[y == II, 1], alpha=.8, color=color)
    show()

def PrepareData(classSize = 10):
    """

    :param classSize:
        For each of the labels, what is the max size we want for each of the labels.
    """
    images, labels = extract_training_samples("byclass")
    Idx = []
    for Label in range(0, 62):
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