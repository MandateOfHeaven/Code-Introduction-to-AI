# classifiers.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import util
from classificationMethod import ClassificationMethod
import numpy as np
from scipy.optimize import fmin_slsqp, fmin_l_bfgs_b


class LinearRegressionClassifier(ClassificationMethod):
    """
    Classifier with Linear Regression.
    """

    def __init__(self, legalLabels):
        """

        :param legalLabels: Labels to predict (for digit data, legalLabels = range(10))
        """
        super(LinearRegressionClassifier, self).__init__(legalLabels)
        self.legalLabels = legalLabels
        self.type = 'lr'
        self.lambda_ = 1e-4
        self.weights = None

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Train the Linear Regression Classifier.

        For digit data, trainingData/validationData are all in numpy format with size ([number of data], 784)
        For doc data, trainingData/validationData should also be in numpy format.
        """
        n, dim = trainingData.shape
        X = trainingData
        Y = np.zeros((n, len(self.legalLabels)))
        Y[np.arange(n), trainingLabels] = 1
        self.weights = np.dot(np.linalg.inv(
            np.dot(X.T, X) + self.lambda_ * np.eye(dim)), np.dot(X.T, Y))

    def classify(self, data):
        """
        Predict which class is in.
        :param data: data to classify which class is in. (in numpy format)
        :return list or numpy array
        """
        return np.argmax(np.dot(data, self.weights), axis=1)


class KNNClassifier(ClassificationMethod):
    """
    KNN Classifier.
    """

    def __init__(self, legalLabels, num_neighbors):
        """

        :param legalLabels: Labels to predict (for digit data, legalLabels = range(10))
        :param num_neighbors: number of nearest neighbors.
        """
        super(KNNClassifier, self).__init__(legalLabels)
        self.legalLabels = legalLabels
        self.type = 'knn'
        self.num_neighbors = num_neighbors

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Train the Linear Regression Classifier by just storing the trainingData and trainingLabels.

        For digit data, trainingData/validationData are all in numpy format with size ([number of data], 784)
        For doc data, trainingData/validationData should also be in numpy format.
        """

        # trainingData is normalized
        self.trainingData = trainingData / \
            np.linalg.norm(trainingData, axis=1).reshape(
                (len(trainingData), 1))
        self.trainingLabels = trainingLabels

    def classify(self, data):
        """
        Predict which class is in.

        Some numpy functions that may be of use (we consider np as short of numpy)
        np.sum(a, axis): sum of array elements over a given axis.
        np.dot(A, B): dot product of two arrays, or matrix multiplication between A and B.
        np.sort, np.argsort: return a sorted copy (or indices) of an array.

        :param data: Data to classify which class is in. (in numpy format)
        :return Determine the class of the given data (list or numpy array)
        """

        data = data / np.linalg.norm(data, axis=1).reshape((len(data), 1))
        "*** YOUR CODE HERE ***"
        # Initialize the data
        n = len(data)
        traindata = self.trainingData
        trainlabels = self.trainingLabels
        datalabels = np.ones((n, 1))
        num_neighbors = self.num_neighbors
        from collections import Counter
        # classify
        for i in range(len(data)):
            # select the knn
            ThisArray = data[i] * np.ones((len(traindata), 1))
            DotResult = ThisArray * traindata
            Distance = np.sum(DotResult, axis=1)
            Indices = np.argsort(-Distance)
            Neighbors_Indices = Indices[0:num_neighbors]
            Neighbors = [trainlabels[j] for j in Neighbors_Indices]
            # choose the most common class
            ThisClass = Counter(Neighbors).most_common(1)[0][0]
            datalabels[i] = ThisClass
        return datalabels


class PerceptronClassifier(ClassificationMethod):
    """
    Perceptron classifier.
    """

    def __init__(self, legalLabels, max_iterations):
        """
        self.weights/self.bias: parameters to train, can be considered as parameter W and b in a perception.
        self.batchSize: batch size in a mini-batch, used in SGD method
        self.weight_decay: weight decay parameters.
        self.learningRate: learning rate parameters.

        :param legalLabels: Labels to predict (for digit data, legalLabels = range(10))
        :param max_iterations: maximum epoches
        """
        super(PerceptronClassifier, self).__init__(legalLabels)
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = 50
        self.weights = None
        self.bias = None
        self.batchSize = 100
        self.weight_decay = 1e-3
        self.learningRate = 1

    def setWeights(self, input_dim):
        self.weights = np.random.randn(input_dim, len(
            self.legalLabels)) / np.sqrt(input_dim)
        self.bias = np.zeros(len(self.legalLabels))

    def prepareDataBatches(self, traindata, trainlabel):
        """
        Generate data batches with given batch size(self.batchsize)

        :return a list in which each element are in format (batch_data, batch_label). E.g.:
            [(batch_data_1), (batch_label_1), (batch_data_2,
              batch_label_2), ..., (batch_data_n, batch_label_n)]

        """
        index = np.random.permutation(len(traindata))
        traindata = traindata[index]
        trainlabel = trainlabel[index]
        split_no = int(len(traindata) / self.batchSize)
        return zip(np.split(traindata[:split_no * self.batchSize], split_no), np.split(trainlabel[:split_no * self.batchSize], split_no))

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        The training loop for the perceptron passes through the training data several
        times and updates the weight vector for each label based on classification errors.
        See the project description for details.

        For digit data, trainingData/validationData are all in numpy format with size ([number of data], 784)
        For doc data, trainingData/validationData should also be in numpy format.

        Some data structures that may be in use:
        self.weights/self.bias (numpy format): parameters to train,
            can be considered as parameter W and b in a perception.
        self.batchSize (scalar): batch size in a mini-batch, used in SGD method
        self.weight_decay (scalar): weight decay parameters.
        self.learningRate (scalar): learning rate parameters.

        Some numpy functions that may be of use (we consider np as short of numpy)
        np.sum(a, axis): sum of array elements over a given axis.
        np.dot(A, B): dot product of two arrays, or matrix multiplication between A and B.
        np.mean(a, axis): mean value of array elements over a given axis
        np.exp(a)
        """

        self.setWeights(trainingData.shape[1])
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.

        # Hyper-parameters. Your can reset them. Default batchSize = 100,
        # weight_decay = 1e-3, learningRate = 1
        "*** YOU CODE HERE ***"
        self.batchSize = 100
        self.weight_decay = 1e-3
        self.learningRate = 1
        weight = self.weights
        bias = self.bias
        eta = float(self.learningRate)
        lamda = self.weight_decay
        k = self.batchSize
        for iteration in range(self.max_iterations):
            print "Starting iteration ", iteration, "..."
            dataBatches = self.prepareDataBatches(trainingData, trainingLabels)
            for batchData, batchLabel in dataBatches:
                "*** YOUR CODE HERE ***"
                # calculate probility
                OutputValue = np.dot(batchData, weight) + bias  # may be error
                temp = (np.sum(np.exp(OutputValue), axis=1)) * np.ones((10, 1))
                temp = temp.T
                prob = np.exp(OutputValue) / temp  # (100,10)

                # update weight and bias

                sum_w = np.zeros((10, 784))
                sum_b = np.zeros((10, 1))
                for i in range(k):
                    ThisProb = prob[i]
                    ThisProb.shape = 10, 1
                    # update Delta_w
                    Delta_w_ini = batchData[i] * ThisProb
                    Delta_w = np.zeros((10, 784))
                    ThisLabel = batchLabel[i]
                    Delta_w[ThisLabel] = -np.ones((1, 784))
                    temp = np.ones((10, 1))
                    Thistemp = batchData[i] * temp
                    Delta_w = Delta_w * Thistemp + Delta_w_ini
                    sum_w = sum_w + Delta_w
                    # updata Delta_b
                    Delta_b = np.zeros((10, 1))
                    Delta_b[ThisLabel] = -1
                    Delta_b = Delta_b + ThisProb
                    sum_b = sum_b + Delta_b
                # update weight and bias
                weight = weight - eta * lamda * \
                    weight - (eta / k) * sum_w.T
                bias = bias - eta * lamda * \
                    bias - (eta / k) * sum_b.T
        # set the value
        self.weights = weight
        self.bias = bias

    def classify(self, data):
        """
        :param data: Data to classify which class is in. (in numpy format)
        :return Determine the class of the given data (list or numpy array)
        """

        return np.argmax(np.dot(data, self.weights) + self.bias, axis=1)

    def visualize(self):
        sort_weights = np.sort(self.weights, axis=0)
        _min = 0
        _max = sort_weights[-10]
        return np.clip(((self.weights - _min) / (_max - _min)).T, 0, 1)


class SVMClassifier(ClassificationMethod):
    """
    SVM Classifier
    """

    def __init__(self, legalLabels, max_iterations=3000, C=1.0, kernelType='rbf'):
        """
        self.sigma: \sigma value in Gaussian RBF kernel.
        self.support/sulf.support_vectors: support vectors and support(y*\alpha). May be in list or dict format.

        :param legalLabels: Labels to predict (for digit data, legalLabels = range(10))
        :param max_iterations: maximum iterations in optimizing constrained QP problem
        :param C: value C in SVM
        :param kernelType: kernel type. Only 'rbf' or 'linear' are valid
        """
        super(SVMClassifier, self).__init__(legalLabels)
        self.type = 'svm'
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.C = C
        self.kernelType = kernelType
        self.sigma = 1.0

        # may be used in training
        self.support = None
        self.support_vectors = None
        self.biases = None

        # DO NOT change self.testing, or you MAY NOT PASS THE AUTOGRADER
        self.testing = False

    def optimizeConstrainedQuad(self, x, A, b, bounds, E, e, debug=False):
        """
        min 1/2 x^T A x + b^T x
        s.t. bounds[i][0] <= x_i <= bounds[i][1]
             E x = e

        :param x : vector of dimension n;
        :param A : matrix of dimension n*n;
        :param: bounds: list of vector, each vector with size 2, length of list is n.
        :param: E: matrix of size m*n
        :param e: vector of size m
        :param debug: whether to output the intermediate results during optimization
        :return optimized x
        """

        if len(E.shape) == 1:
            E = E.reshape((1, E.shape[0]))
            e = np.array(e).reshape(1)

        assert x.shape[0] == A.shape[0] and x.shape[0] == A.shape[1]
        assert x.shape[0] == E.shape[1]
        assert x.shape[0] == len(bounds)
        assert sum(len(bnd) == 2 for bnd in bounds) == len(bounds)
        assert E.shape[0] == e.shape[0]

        if self.testing:
            np.savez('test_cases/student_test_cqp.npz', A=np.array(A),
                     b=np.array(b), bounds=np.array(bounds), E=np.array(E), e=np.array(e))

        n = x.shape[0]
        func = lambda x: 0.5 * np.dot(x, np.dot(A, x)) + np.dot(b, x)
        f_eqcons = lambda x: np.dot(E, x) - e
        bounds = bounds
        fprime = lambda x: np.dot(A, x) + b
        fprime_eqcons = lambda x: E
        func_w_eqcon = lambda x: func(x) + n * 5e-5 * np.sum(f_eqcons(x)**2)
        fprime_w_eqcon = lambda x: fprime(
            x) + n * 1e-4 * np.dot(f_eqcons(x), fprime_eqcons(x))
        max_iters = self.max_iterations
        iprint = 90 if debug else 0
        res = fmin_l_bfgs_b(func_w_eqcon, x, fprime=fprime_w_eqcon, bounds=bounds,
                            maxfun=max_iters, maxiter=max_iters, factr=1e10, iprint=iprint)
        print 'F = %.4f Eqcons Panelty = %.4f' % (func(res[0]), np.sum(f_eqcons(res[0])**2))
        return res[0]

    def generateKernelMatrix(self, data1, data2=None):
        """
        Generate a kernel. Linear Kernel and Gaussian RBF Kernel is provided.

        :param data1: in numpy format
        :param data2: in numpy format
        :return:
        """
        if data2 is None:
            data2 = data1
        if self.kernelType == 'rbf':
            X12 = np.sum(data1 * data1, axis=1, keepdims=True)
            X22 = np.sum(data2 * data2, axis=1, keepdims=True)
            XX = 2 * np.dot(data1, data2.T) - X12 - X22.T
            XX = np.exp(XX / (2 * self.sigma * self.sigma))
        elif self.kernelType == 'linear':
            XX = np.dot(data1, data2.T)
        else:
            raise Exception('Unknown kernel type: ' + str(self.kernelType))
        return XX

    def trainSVM(self, trainData, trainLabels):
        """
        Train SVM with just two labels: 1 and -1

        :param traindata: in numpy format
        :param trainLabels: in numpy format

        Some functions that may be of use:
        self.optimizeConstrainedQuad: solve constrained quadratic programming problem
        self.generateKernelMatrix: Get kernel matrix given specific type of kernel
        """
        assert len(trainData) == len(trainLabels)
        assert (np.sum(trainLabels == 1) +
                np.sum(trainLabels == -1)) == len(trainLabels)
        alpha = np.zeros(len(trainData))
        KernelMat = self.generateKernelMatrix(trainData)

        "*** YOUR CODE HERE ***"
        x = alpha
        y = trainLabels.reshape((len(trainLabels), 1))
        y = y.T * y
        A = y * KernelMat
        b = -np.ones(len(alpha))
        bounds = [(0, self.C) for i in range(len(alpha))]
        E = trainLabels
        e = 0
        alpha = self.optimizeConstrainedQuad(x, A, b, bounds, E, e)
        return alpha

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        ovr(one vs. the rest) training with SVM

        For digit data, trainingData/validationData are all in numpy format with size ([number of data], 784)
        For doc data, trainingData/validationData should also be in numpy format.

        Some numpy functions that may be of use (we consider np as short of numpy)
        np.where(condition, x, y): Return elements, either from x or y, depending on condition.
        np.mean(a, axis): mean value of array elements over a given axis
        indexing, slicing in numpy may be important
        """

        # determine sigma with training data
        # import sklearn
        X = trainingData[:1000]
        X2 = np.sum(X * X, axis=1, keepdims=True)
        self.sigma = np.sqrt(np.mean(-2 * np.dot(X, X.T) + X2 + X2.T))

        self.support = {}
        self.support_vectors = {}
        self.biases = {}
        for t in self.legalLabels:
            # for each class, use ovr to train SVM classifier
            print 'classify label', t, '...'
            traindata = trainingData
            trainlabels = np.where(trainingLabels == t, 1, -1)
            # To avoid the precision loss underlying the floating point,
            # we recommend use (alpha > 1e-6 + beta) to determine whether alpha is greater than beta,
            # and (alpha < beta - 1e-6) to determine whether alpha is smaller than beta,
            # and (abs(alpha - beta) < 1e-6) to determine wheter alpha is equal
            # to beta.
            "*** YOUR CODE HERE ***"
            alpha = self.trainSVM(traindata, trainlabels)
            recorder = 0
            support = np.zeros((len(alpha), 1))
            support_vector = np.zeros((traindata.shape))
            bias = np.zeros((len(alpha), 1))
            Delta = 1e-6
            for i in range(len(alpha)):
                if alpha[i] > Delta:
                    support[i] = alpha[i] * trainlabels[i]
                    support_vector[i] = traindata[i]
            KernelMat = self.generateKernelMatrix(support_vector, traindata)
            for i in range(len(alpha)):
                if alpha[i] > Delta and alpha[i] < self.C - Delta:
                    bias[i] = trainlabels[i] - np.dot(KernelMat[:, i], support)
                    recorder = recorder + 1
            bias = np.sum(bias) / recorder
            # util.raiseNotDefined()
            self.support[t] = support
            self.support_vectors[t] = support_vector
            self.biases[t] = bias

    def classify(self, data):
        """
        ovr(one vs. the rest) classification with SVM
        """
        "*** YOUR CODE HERE ***"
        result = np.zeros((len(data), len(self.legalLabels)))
        for t in self.legalLabels:
            i = int(t)
            KernelMat = self.generateKernelMatrix(
                data, self.support_vectors[t])
            A = KernelMat * self.support[t].T
            f = np.sum(A, axis=1) + self.biases[t]
            result[:, i] = f
        FinalClass = np.argmax(result, axis=1)
        return FinalClass
        # util.raiseNotDefined()
