from utils.core import *


def LDADemonstration():
    LdaInstance = LDADimReduce();
    print(f"Geting the LDA Model... ")
    Data, Labels = SplitbyClasses(classSize=1000, classes=[26, 27, 28])
    # new data that never seemed before.
    Embeddings = LdaInstance.getEmbeddings(Data);
    print("Getting Embeddings...")
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

    # LDADemonstration()


def PCAPlusLDADemonstration():
    PCAInstance = PCADimReduce(n_components=0.8);
    print("Getting PCA Model...")
    PCAEmbeddings = PCAInstance.getEmbeddings();
    print("Getting PCA Embeddings...")
    TrainLabels = PCAInstance.Labels
    print("Train LDA on PCA Embeddings... ")
    LDAMaxComponents = min(61, PCAInstance.PcaModel.n_components_ - 1)
    LDAInstance = LDADimReduce(n_components=LDAMaxComponents,  # Just to be sure.
                               X=PCAEmbeddings,
                               y=TrainLabels)  # Use the PCA embeddings to train LDA

    classes = [SymbolsToLabels(Char) for Char in "0o2z"]
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

    # PCAPlusLDADemonstration()


def SVMTesting():
    DisplayLabels = "abcdefghijklmnopqrstuvwxyz"
    classes = [SymbolsToLabels(II) for II in DisplayLabels]
    Model = make_pipeline(StandardScaler(), SVC(gamma="auto"))  # Making the SVC Model.

    TrainX, TrainY = SplitbyClasses(classes=classes, classSize=1000)
    TestX, TestY = SplitbyClasses(classes=classes, classSize=3000)

    DimRe = LDADimReduce(X=TrainX, y=TrainY)  # Use Train set to create LDA embeddings.
    TrainEmbeddings = DimRe.getEmbeddings()  # Get Embeddings from the set trained LDA
    TestEmbeddings = DimRe.getEmbeddings(TestX)  # Get the embeddings from the test set.

    Model.fit(TrainEmbeddings, TrainY);
    print(f"Modeling Has been fitted: ")
    print(Model)
    disp = plot_confusion_matrix(Model, TestEmbeddings, TestY, display_labels=list(DisplayLabels))
    disp.ax_.set_title("Test Set SVM")
    disp = plot_confusion_matrix(Model, TrainEmbeddings, TrainY, display_labels=list(DisplayLabels))
    disp.ax_.set_title("Training Set SVM")
    show()


def SplitByClassesShuffleTesting():
    classes = list(range(3))
    X1, _ = SplitbyClasses(shuffle_data=True, classSize=10, classes=classes)
    X2, _ = SplitbyClasses(shuffle_data=True, classSize=10, classes=classes)
    print("Making sure the the suffle data are not the same")
    assert not array_equal(X1, X2)
    print("Yes, if you see this ten it's assserted and they are not equal. ")
    X1, _ = SplitbyClasses(shuffle_data=False, classSize=10, classes=classes)
    X2, _ = SplitbyClasses(shuffle_data=False, classSize=10, classes=classes)
    print("If not shuffled, they should come out and be the same")
    assert array_equal(X1, X2)
    print("Yes, if you see this then the condition has been asserted. ")


def TestConfusionMatrix():
    FakeLabels = list(range(30))
    FakeLabels2 = list(range(30))
    MyConMat = ConfusionMatrix(FakeLabels, FakeLabels2)
    print(MyConMat.TotalAccuracy)
    print(MyConMat.FalseNegativeEach)
    print(MyConMat.FalsePositiveEach)
    fig, ax = MyConMat.report()
    fig, ax = MyConMat.visualize()
    show()
    fig.show()


def TestDecisionTreeEnsemble():
    Symbols = "0o2zZO"
    classes = [SymbolsToLabels(II) for II in Symbols]
    TrainX, TrainY = SplitbyClasses(classes=classes, classSize=3000)
    TestX, TestY = SplitbyClasses(classes=classes, classSize=3000, shuffle_data=True, test_set=True)
    DimRe = LDADimReduce(X=TrainX, y=TrainY)  # Use Train set to create LDA embeddings.
    TrainEmbeddings = DimRe.getEmbeddings()  # Get Embeddings from the set trained LDA
    TestEmbeddings = DimRe.getEmbeddings(TestX)  # Get the embeddings from the test set.
    Model = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    Model.fit(TrainEmbeddings, TrainY)
    Score = Model.score(TestEmbeddings, TestY)
    print(f"Random Forest Score Test set: {Score}")
    Score = Model.score(TrainEmbeddings, TrainY)
    print(f"Trandom Forest Score Train set: {Score}")


def TestAdaBoost():
    Symbols = "0123456789"
    classes = [SymbolsToLabels(II) for II in Symbols]
    Model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                               algorithm="SAMME",
                               n_estimators=200)
    TrainX, TrainY = SplitbyClasses(classes=classes, classSize=3000)
    TestX, TestY = SplitbyClasses(classes=classes, classSize=200, shuffle_data=True, test_set=True)
    DimRe = DimReduceHybrid(X=TrainX, y=TrainY)  # Use Train set to create LDA embeddings.
    TrainEmbeddings = DimRe.getEmbeddings()  # Get Embeddings from the set trained LDA
    TestEmbeddings = DimRe.getEmbeddings(TestX)  # Get the embeddings from the test set.
    Model.fit(TrainEmbeddings, TrainY)
    print(f"AdaBoost Test Score {Model.score(TestEmbeddings, TestY)}")
    print(f"AdaBoost Train Score {Model.score(TrainEmbeddings, TrainY)}")

def TestLabelsorganizer():
    classes = [SymbolsToLabels(S) for S in "aA"]
    DataOrganizer = LabelsOrganizer(classSize=3, shuffle_data=True, classes=classes)
    Data, Labels = DataOrganizer.getDataLabels()
    matplotlib.rcParams['figure.figsize'] = (3, 3)
    for II, D in np.ndenumerate(Labels):
        plt.imshow(reshape(Data[II,:], (28, 28)))
        plt.title(f"The label is: {D}")
        show()

def TestSplitByClasses():
    classes = [SymbolsToLabels(S) for S in "aAbB"]
    Data, Labels = SplitbyClasses(classSize=3, shuffle_data=True, classes = classes)
    matplotlib.rcParams['figure.figsize'] = (3, 3)
    for II, D in np.ndenumerate(Labels):
        plt.imshow(reshape(Data[II, :], (28, 28)))
        plt.title(f"The label is: {D}")
        show()


def main():
    TestSplitByClasses()
    pass

if __name__ == "__main__":
    import os
    print(f"Script running with cwd: {os.getcwd()}")
    main()
