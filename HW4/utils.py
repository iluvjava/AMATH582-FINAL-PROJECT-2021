import numpy as np
import matplotlib.pyplot as plt
import scipy
import gzip
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import svm
from sklearn import tree
from sklearn.metrics import accuracy_score


def load_images(dataset):
    with gzip.open(dataset, "r") as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), "big")
        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), "big")
        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), "big")
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), "big")
        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8).reshape(
            (image_count, row_count, column_count)
        )
        return images


def load_labels(dataset):
    with gzip.open(dataset, "r") as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), "big")
        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), "big")
        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels


def prep_data(n_digits, digits, n_features=10):
    if n_digits != len(digits):
        print("Unmatched digits, check input")
        return 0
        pass
    print(n_features)

    global train_X_proj, test_X_proj
    global train_labels, test_labels

    if n_digits == 2:
        digit1 = digits[0]
        digit2 = digits[1]

        train_arg1 = np.argwhere(train_labels == digit1)
        train_arg2 = np.argwhere(train_labels == digit2)

        train_X1 = train_X_proj[train_arg1[:, 0], 1:n_features]
        train_X2 = train_X_proj[train_arg2[:, 0], 1:n_features]

        train_y1 = train_labels[train_arg1[:, 0]]
        train_y2 = train_labels[train_arg2[:, 0]]

        test_arg1 = np.argwhere(test_labels == digit1)
        test_arg2 = np.argwhere(test_labels == digit2)

        test_X1 = test_X_proj[test_arg1[:, 0], 1:n_features]
        test_X2 = test_X_proj[test_arg2[:, 0], 1:n_features]

        test_y1 = test_labels[test_arg1[:, 0]]
        test_y2 = test_labels[test_arg2[:, 0]]

        train_X_digits = np.concatenate((train_X1, train_X2), axis=0)
        train_y_digits = np.concatenate((train_y1, train_y2), axis=0)

        test_X_digits = np.concatenate((test_X1, test_X2), axis=0)
        test_y_digits = np.concatenate((test_y1, test_y2), axis=0)

    if n_digits == 3:
        digit1 = digits[0]
        digit2 = digits[1]
        digit3 = digits[2]

        train_arg1 = np.argwhere(train_labels == digit1)
        train_arg2 = np.argwhere(train_labels == digit2)
        train_arg3 = np.argwhere(train_labels == digit3)

        train_X1 = train_X_proj[train_arg1[:, 0], 1:n_features]
        train_X2 = train_X_proj[train_arg2[:, 0], 1:n_features]
        train_X3 = train_X_proj[train_arg3[:, 0], 1:n_features]

        train_y1 = train_labels[train_arg1[:, 0]]
        train_y2 = train_labels[train_arg2[:, 0]]
        train_y3 = train_labels[train_arg3[:, 0]]

        test_arg1 = np.argwhere(test_labels == digit1)
        test_arg2 = np.argwhere(test_labels == digit2)
        test_arg3 = np.argwhere(test_labels == digit3)

        test_X1 = test_X_proj[test_arg1[:, 0], 1:n_features]
        test_X2 = test_X_proj[test_arg2[:, 0], 1:n_features]
        test_X3 = test_X_proj[test_arg3[:, 0], 1:n_features]

        test_y1 = test_labels[test_arg1[:, 0]]
        test_y2 = test_labels[test_arg2[:, 0]]
        test_y3 = test_labels[test_arg3[:, 0]]

        train_X_digits = np.concatenate((train_X1, train_X2, train_X3), axis=0)
        train_y_digits = np.concatenate((train_y1, train_y2, train_y3), axis=0)

        test_X_digits = np.concatenate((test_X1, test_X2, test_X3), axis=0)
        test_y_digits = np.concatenate((test_y1, test_y2, test_y3), axis=0)

    return train_X_digits, train_y_digits, test_X_digits, test_y_digits
    pass


def make_bar_plot(y_predict, n_digits, digits):
    fig = plt.figure()
    plt.plot(y_predict)
    plt.ylabel("prediction results")
    plt.xlabel("test datasets")
    if n_digits == 2:
        plt.savefig(f"{digits[0]}_vs_{digits[1]}.pdf")
    if n_digits == 3:
        plt.savefig(f"{digits[0]}_vs_{digits[2]}_vs_{digits[3]}.pdf")

    pass


def train_model(model, n_digits, digits, n_features=10, do_cv=False, do_plot=False):
    [train_X, train_y, test_X, test_y] = prep_data(n_digits, digits, n_features)

    if model == "lda":
        model = LDA()
    elif model == "svm":
        model = svm.SVC()
    elif model == "tree":
        model = tree.DecisionTreeClassifier()

    if do_cv:
        cv_score = cross_val_score(model, train_X, train_y, cv=10, scoring="accuracy")
        mean_cv_score = np.mean(cv_score)
        print(f"cross_val_score:\n", cv_score)
        print(f"mean cross_val_score: ", mean_cv_score)

        return mean_cv_score

    model.fit(train_X, train_y)
    y_predict = model.predict(test_X)
    accu = accuracy_score(test_y, y_predict)

    print(f"the accuracy on test dataset: {accu}")

    if do_plot:
        make_bar_plot(y_predict, n_digits, digits)

    return accu

    pass
