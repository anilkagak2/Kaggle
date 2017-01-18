import numpy as np
import pickle
from scipy.ndimage import imread
from scipy.misc import imresize
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import os
import cv2

'''
Insights
1) There are many objects in the background and it would really help the algo to just get rid of them
2) Mostly there's water, human beings, containers, etc. These are colorful. Check if just removing them helps.
3) Are fishes mostly black-white-silver and red in color? But there's whale which is blue
4) Fish will be very small in the picture, other objects are relatively large and makes the pic.
'''

#newShape = (640, 480)
newShape = (60, 40)

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

def GatherTrainTestAndEvaluate(Data_Dir):
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

def experimentOnAnImage(Data_Dir):
    print(Data_Dir)
    label = 'ALB'
    print(label)
    for root, dirs, files in os.walk(os.path.join(Data_Dir, label)):
        print(root)
        for name in files:
            print((os.path.join(root, name)))
            img = imread(os.path.join(root, name)) 
            #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            #gray = np.float32(gray)
            #dst = cv2.cornerHarris(gray,2,3,0.04)

            #result is dilated for marking the corners, not important
            #dst = cv2.dilate(dst,None)

            # Threshold for an optimal value, it may vary depending on the image.
            #img[dst>0.01*dst.max()]=[0,0,255]

            #cv2.imshow('dst',img)
            #cv2.imshow('img', img)

            #sift = cv2.xfeatures2d.SIFT_create()
            #kp = sift.detect(gray,None)
            #img=cv2.drawKeypoints(gray,kp,img)
            #cv2.imshow('sift_keypoints.jpg',img)
            cv2.imshow('sift_keypoints.jpg',img)

            cv2.waitKey()
            #break

if __name__ == '__main__':
    Data_Dir = 'C:\\Users\\t-anik\\Desktop\\personal\\KaggleData'

    GatherTrainTestAndEvaluate(Data_Dir)
    #experimentOnAnImage(os.path.join(Data_Dir, 'train'))
