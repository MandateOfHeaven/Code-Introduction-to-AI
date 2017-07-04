# sentimentAnalysis.py

# imports
import util
from featureExtractor import BaseFeatureExtractor
from classificationMethod import ClassificationMethod
import numpy as np


vocabularyList = {}  # vocabulary list
vocabularyCounts = {}  # The number of occurrences of a word in SSD sentences
with open('data/SST/vocabulary.txt', 'r') as f:
    i = 0
    for line in f:
        v, c = line.strip().split()
        vocabularyCounts[v] = int(c)
        vocabularyList[v] = i
        i += 1


def loadTextData():
    rawTrainingData = []
    rawValidationData = []
    rawTestData = []
    with open('data/SST/trainingData.txt', 'r') as f:
        for line in f:
            rawTrainingData.append(line.strip())
    with open('data/SST/validationData.txt', 'r') as f:
        for line in f:
            rawValidationData.append(line.strip())
    with open('data/SST/testData.txt', 'r') as f:
        for line in f:
            rawTestData.append(line.strip())
    with open('data/SST/trainingLabels.txt', 'r') as f:
        rawTrainingLabels = [int(t) for t in f.read().strip().split()]
    with open('data/SST/validationLabels.txt', 'r') as f:
        rawValidationLabels = [int(t) for t in f.read().strip().split()]
    with open('data/SST/testLabels.txt', 'r') as f:
        rawTestLabels = [int(t) for t in f.read().strip().split()]
    return rawTrainingData, rawTrainingLabels, rawValidationData, rawValidationLabels, rawTestData, rawTestLabels


class FeatureExtractorText(BaseFeatureExtractor):
    """
    Extract text data given a list of sentences.
    """

    def __init__(self):
        super(FeatureExtractorText, self).__init__()

    def fit(self, trainingData):
        """
        Train feature extractor given the text training Data (not in numpy format)
        :param trainingData: a list of sentences
        :return:
        """

        "*** YOUR CODE HERE ***"

        self.count_vect = None
        self.tf_transformer = None

        # Tokenizing text with sklearn
        from sklearn.feature_extraction.text import CountVectorizer
        self.count_vect = CountVectorizer()
        X_train_counts = self.count_vect.fit_transform(trainingData)

        # From occurrences to frequencies
        from sklearn.feature_extraction.text import TfidfTransformer
        self.tf_tranformer = TfidfTransformer(
            use_idf=False).fit(X_train_counts)
        X_train_tf = self.tf_tranformer.transform(X_train_counts)

    def extract(self, data):
        """
        Extract the feature of text data
        :param data: a list of sentences (not in numpy format)
        :return: features, in numpy format and len(features)==len(data)
        """
        "*** YOUR CODE HERE ***"
        X_train_counts = self.count_vect.transform(data)
        # From occurrences to frequencies
        X_train_tf = self.tf_tranformer.transform(X_train_counts)
        return X_train_tf

    def visualize(self, data):
        """
        May be used for ease of visualize the text data
        :param data:
        :return:
        """
        pass


class ClassifierText(ClassificationMethod):
    """
    Perform classification to text data set
    """

    def __init__(self, legalLabels):
        super(ClassifierText, self).__init__(legalLabels)
        # You may use the completed classification methods
        # or directly use sklearn for learning
        # e.g.
        # import classifiers
        # self.classifier = classifiers.PerceptronClassifier(legalLabels, 50)
        "*** YOUR CODE HERE ***"
        self.legalLabels = legalLabels
        self.tf_transformer = None
        self.trainedclassifier = {}

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Train the text classifier with text features
        :param trainingData: in numpy format
        :param trainingLabels: in numpy format
        :param validationData: in numpy format
        :param validationLabels: in numpy format
        """

        # You may use the completed classification methods
        # e.g.
        # self.classifier.train(trainingData, trainingLabels, validationData,
        # validationLabels)
        "*** YOUR CODE HERE ***"

        # train the classifiers, including
        import sklearn.naive_bayes as nb
        import sklearn.svm as svm
        # MultinomialNB, 39.5%, 40.0%
        clf = nb.MultinomialNB(alpha=0.191).fit(trainingData, trainingLabels)
        self.trainedclassifier['MultinomialNB'] = clf

        # BernoulliNB, 37.8% 40.4%
        clf = nb.BernoulliNB(alpha=0.3).fit(trainingData, trainingLabels)
        self.trainedclassifier['BernoulliNB'] = clf

        # LinearSVC, 37.0% 40.9%
        clf = svm.LinearSVC(C=0.25, tol=0.0001,  dual=False, penalty='l1').fit(
            trainingData, trainingLabels)
        self.trainedclassifier['LinearSVC'] = clf

    def classify(self, data):
        """
        Classify the text classifier with text features
        :param data: in numpy format
        :return:
        """

        # You may use the completed classification methods
        # e.g.
        # return self.classifier.classify(data)
        "*** YOUR CODE HERE ***"

        from collections import Counter
        predicted = np.zeros((data.shape[0], 3))
        recorder = 0
        for t in self.trainedclassifier:
            predicted[:, recorder] = self.trainedclassifier[t].predict(data)
            recorder = recorder + 1
        FinalResult = np.zeros((data.shape[0], 1))
        ThreeClass = np.zeros((data.shape[0], 3))
        WholeBias = np.zeros((data.shape[0], 1))
        for i in range(len(FinalResult)):
            for j in range(3):
                if predicted[i, j] > 2:
                    ThreeClass[i, j] = 1
                elif predicted[i, j] < 2:
                    ThreeClass[i, j] = -1
                elif predicted[i, j] == 2:
                    ThreeClass[i, j] = 0
            WholeBias[i] = np.mean(ThreeClass[i])

        for i in range(len(FinalResult)):
            if predicted[i, 0] == predicted[i, 1]:
                FinalResult[i] = predicted[i, 0]
            elif predicted[i, 0] == predicted[i, 2]:
                FinalResult[i] = predicted[i, 0]
            elif predicted[i, 1] == predicted[i, 2]:
                FinalResult[i] = predicted[i, 1]
            elif WholeBias[i] > 0:
                FinalResult[i] = 3
            elif WholeBias[i] < 0:
                FinalResult[i] = 1
            elif WholeBias[i] == 0:
                FinalResult[i] = 2
        return FinalResult
