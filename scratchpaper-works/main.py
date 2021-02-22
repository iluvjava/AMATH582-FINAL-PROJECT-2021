CURRENT_DIRECTORY = None

from emnist import list_datasets, extract_training_samples
import sys, os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from collections import Counter

def main():
    images, labels = extract_training_samples("byclass");
    plt.imshow(images[0, :, :])
    plt.show()
    Counts = Counter(labels)
    print("Label Counter: ")
    print(Counts)

    print("Sorting")
    PermuteVec = np.argsort(labels)
    SortedImages = np.zeros(images.shape, dtype=np.int8)
    SortedLabels = np.zeros(labels.shape, dtype=np.int8)
    print("Permutating...")
    for Idx, Val in enumerate(PermuteVec):
        SortedLabels[Idx] = labels[Val]
        SortedImages[Idx, :, :] = images[Val, :, :]
    print("Saving")
    sio.savemat("images.mat", {"images": SortedImages})
    sio.savemat("labels.mat", {"labels": SortedLabels})


if __name__ == "__main__":
    if CURRENT_DIRECTORY is not None:
        os.chdir(CURRENT_DIRECTORY)
    print(f"Script CWD: {os.getcwd()}")
    main()