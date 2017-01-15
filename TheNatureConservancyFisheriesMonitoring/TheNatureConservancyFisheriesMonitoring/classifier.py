import numpy as np
import pickle
from scipy.ndimage import imread
from scipy.misc import imresize
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import os

newShape = (640, 480)

def cleanImage(im):
    global newShape
    im = imresize(im, newShape)
    return im.flatten()

def get_features_and_labels(data_dir):
    classLabels = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    labels = []
    data = []
    for i in range(len(classLabels)):
        label = classLabels[i]
        print(label)
        for root, dirs, files in os.walk(os.path.join(data_dir, label)):
            for name in files:
                #print((os.path.join(root, name)))
                data.append( cleanImage( imread(os.path.join(root, name)) ) )
                labels.append(i)

    data = np.array(data)
    labels = np.array(labels)
    print(data.shape)
    print(labels.shape)
    print(data[0].shape)
    print(labels[0].shape)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def saveModel(model, modelName):
    pickle.dump( model, open( modelName, "wb" ) )

if __name__ == '__main__':
    Data_Dir = 'C:\\Users\\t-anik\\Desktop\\personal\\KaggleData'

    # Process data into feature and label arrays
    X_train, X_test, y_train, y_test = get_features_and_labels(os.path.join(Data_Dir, 'train'))

    # Train the classifier
    #clf = svm.SVC()
    clf = svm.LinearSVC()
    clf.fit(X_train, y_train)

    # Predict on the test set and report the metrics
    predicted = clf.predict(X_test)
    print(metrics.classification_report(y_test, predicted))

    # Save the classifier
    saveModel(clf, "model-svc-default.bin")

