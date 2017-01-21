from __future__ import print_function
import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T

import numpy as np
import pickle
from scipy.ndimage import imread
from scipy.misc import imresize
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
import cv2


#newShape = (60, 40)
newShape = (20, 20)
modelName = "model-svc-default.bin"
predictionsFilename = "predictions-SingleNeuralNet-SGD_60_x_40.csv"
classLabels = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):

        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                            dtype=theano.config.floatX),
                                borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                            dtype=theano.config.floatX),
                                borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

def load_data(dataset):
    print('... loading data')
    X_train, X_test, y_train, y_test = get_features_and_labels(os.path.join(Data_Dir, 'train'))

    #y_train = y_train.reshape(y_train.shape[0],1)
    #y_test = y_test.reshape(y_test.shape[0],1)

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.
    test_set_x, test_set_y = shared_dataset((X_test, y_test))
    train_set_x, train_set_y = shared_dataset((X_train, y_train))
    valid_set_x, valid_set_y = shared_dataset((X_test, y_test))

    print("y_train.shape = " + str(y_train.shape))
    print("x_train.shape = " + str(X_train.shape))
    print("shared train y shape = " + str(train_set_y.shape.eval()))
    print("shared train x shape = " + str(train_set_x.shape.eval()))

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    #sys.exit(1)
    return rval


def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
                           dataset='mnist.pkl.gz',
                           batch_size=50):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    print(train_set_x.shape)
    print(train_set_y.shape)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    global newShape 
    global classLabels 

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(input=x, n_in=newShape[0] * newShape[1] * 3, n_out=len(classLabels))

    print("n_in = " + str(newShape[0] * newShape[1]*3))
    print("n_out = " + str(len(classLabels)))

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    #theano.pprint(train_model)
    # end-snippet-3

    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model
                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)

def cleanImage(im):
    global newShape
    im = imresize(im, newShape)
    return im.flatten()

def translateImage(image, dx, dy):
    trans_mat = np.float32([[1,0,dx],[0,1,dy]])
    result = cv2.warpAffine(image, trans_mat, image.shape[:2],flags=cv2.INTER_LINEAR)
    return result

def rotateImage(image, angle):
    center = tuple(np.array(image.shape)[:2]/2) 
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[:2],flags=cv2.INTER_LINEAR)
    return result

def getImageTransformations(im):
    images = []
    images.append(rotateImage(im, 90))
    images.append(rotateImage(im, 180))
    images.append(rotateImage(im, 270))
    images.append(translateImage(im, 0,5))
    images.append(translateImage(im, 5,0))
    images.append(translateImage(im, 0,-5))
    images.append(translateImage(im, -5,0))
    images = [ cleanImage(x) for x in images ]
    return images

def get_features_and_labels(data_dir):
    global classLabels 
    labels = []
    data = []
    for i in range(len(classLabels)):
        label = classLabels[i]
        print(label)
        cnt = 0
        for root, dirs, files in os.walk(os.path.join(data_dir, label)):
            for name in files:
                #print((os.path.join(root, name)))
                img = imread(os.path.join(root, name))
                data.append( cleanImage( img ) )
                labels.append(i)

                for x in getImageTransformations(img):
                    data.append(x)
                    labels.append(i)

                cnt += 1
                if cnt > 100 : break

    data = np.array(data)
    labels = np.array(labels)
    print(data.shape)
    print(labels.shape)
    print(data[0].shape)
    print(labels[0].shape)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)
    return X_train, X_test, y_train, y_test

def get_feature_test_points(data_dir):
    data = []
    filenames = []
    for root, dirs, files in os.walk(data_dir):
        for name in files:
            #print((os.path.join(root, name)))
            data.append( cleanImage( imread(os.path.join(root, name)) ) )
            filenames.append(name)
            #break
    
    data = np.array(data)
    return data, filenames

def writePredictionsToCsv(Data_Dir, predictions, filenames):
    global classLabels
    global predictionsFilename
    import csv
    with open( os.path.join(Data_Dir, predictionsFilename), 'w') as fp:
        writer = csv.writer(fp, lineterminator='\n')
        writer.writerow(["image"] + classLabels)
        for i in range(len(filenames)):
            writer.writerow([filenames[i]] + [ str(x) for x in predictions[i]])

def GatherTestDataAndPredict(Data_Dir):
    print("Prediction..")
    X_test, filenames = get_feature_test_points(os.path.join(Data_Dir, 'test_stg1'))
    
    test_set_x = theano.shared(numpy.asarray(X_test, dtype=theano.config.floatX), borrow=True)
    test_set_x = test_set_x.get_value()

    # load the saved model
    classifier = pickle.load(open('best_model.pkl', 'rb'))

    # compile a predictor function
    #predict_model = theano.function(
    #    inputs=[classifier.input],
    #    outputs=classifier.y_pred)

    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.p_y_given_x)

    predictions = predict_model(test_set_x)
    print("Predicted values for the first 10 examples in test set:")
    #print(predictions)

    #predictions = clf.predict_proba(X_test)
    #predictions = clf.decision_function(X_test) #clf.predict_proba(X_test)

    from sklearn.preprocessing import normalize
    #predictions = normalize(1.0/( 1+np.exp(-1*predictions)), axis=1, norm='l1')
    #print(predictions[0])

    writePredictionsToCsv(Data_Dir, predictions, filenames)


from enum import Enum
class ClassifierStage(Enum):
    Train = 1
    Test = 2
    TrainTest = 3

if __name__ == '__main__':
    Data_Dir = 'C:\\Users\\t-anik\\Desktop\\personal\\KaggleData'

    Stage = ClassifierStage.Test
    if Stage==ClassifierStage.Train or Stage==ClassifierStage.TrainTest : sgd_optimization_mnist(dataset=Data_Dir)
    if Stage==ClassifierStage.Test or Stage==ClassifierStage.TrainTest : GatherTestDataAndPredict(Data_Dir)