import numpy as np
import pickle
from scipy.ndimage import imread
from scipy.misc import imresize
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
import cv2
import sys

'''
Insights
1) There are many objects in the background and it would really help the algo to just get rid of them
2) Mostly there's water, human beings, containers, etc. These are colorful. Check if just removing them helps.
3) Are fishes mostly black-white-silver and red in color? But there's whale which is blue
4) Fish will be very small in the picture, other objects are relatively large and makes the pic.
'''

newShape = (60, 40)
#newShape = (28, 28)
#newShape = (64, 64)
#newShape = (128, 128)
modelName = "model-svc-default.bin"
predictionsFilename = "predictions-RandomForestClassifier_moreTrainData_60_x_40_2_test_stg2.csv"
#predictionsFilename = "predictions-2CNN_1FC_64_64_3.csv"
classLabels = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

def preprocessImage(im):
    global newShape
    #im = imresize(im, newShape)
    im = cv2.resize(im.astype('uint8'), dsize=newShape)
    im = im.astype(np.float)
    return im

def cleanImage(im):
    global newShape
    im = imresize(im, newShape)
    im = im.astype(np.float)
    im = im/ 256.0
    #print(im.flatten()) 
    #sys.exit(1)
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
    
    #angles = [i for i in range(10,360,10)]
    angles = [30, 60, 90, 150, 180, 270]
    for angle in angles:
           images.append(rotateImage(im, angle))

    #dxy = [(0,5), (5,0), (0,10), (10,0), (0,20),(20,0)]
    dxy = [(0,5), (5,0), (0,10), (10,0)]
    for dx, dy in dxy:
        images.append(translateImage(im, dx,dy))
        images.append(translateImage(im, -dx,-dy))

    images = [ cleanImage(x) for x in images ]
    
    return images

def get_images_labels(data_dir):
    global classLabels 
    labels = []
    data = []
    for i in range(len(classLabels)):
        label = classLabels[i]
        print(label)
        
        for root, dirs, files in os.walk(os.path.join(data_dir, label)):
            cnt = 0
            for name in files:
                print((os.path.join(root, name)))
                img = imread(os.path.join(root, name))
                #cv2.imshow('dst',img); cv2.waitKey();
                #img = imread(os.path.join(root, name), mode='L')
                img = preprocessImage(img)
                data.append( img )
                labels.append(i)

                #cv2.imshow('dst',img.astype(np.uint8)); cv2.waitKey();
                '''
                angles = [90, 180, 270]
                for angle in angles:
                    #cv2.imshow('dst',rotateImage(img, angle).astype(np.uint8)); cv2.waitKey();
                    data.append(rotateImage(img, angle)); labels.append(i)
                '''
                '''
                dxy = [(0,5), (5,0)]
                for dx, dy in dxy:
                    #cv2.imshow('dst',translateImage(img, dx,dy).astype(np.uint8)); cv2.waitKey();
                    #cv2.imshow('dst',translateImage(img, -dx,-dy).astype(np.uint8)); cv2.waitKey();
                    data.append(translateImage(img, dx,dy)); labels.append(i)
                    data.append(translateImage(img, -dx,-dy)); labels.append(i)
                '''

                cnt += 1
                if cnt>600: break

    data = np.array(data)
    labels = np.array(labels)
    print(data.shape)
    print(labels.shape)
    print(data[0].shape)
    print(labels[0].shape)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    return X_train, X_test, X_valid, y_train, y_test, y_valid

def get_features_and_labels(data_dir):
    global classLabels 
    labels = []
    data = []
    for i in range(len(classLabels)):
        label = classLabels[i]
        print(label)
        
        for root, dirs, files in os.walk(os.path.join(data_dir, label)):
            cnt = 0
            for name in files:
                print((os.path.join(root, name)))
                img = imread(os.path.join(root, name))
                #img = imread(os.path.join(root, name), mode='L')
                data.append( cleanImage( img ) )
                labels.append(i)
                #break

                '''
                for x in getImageTransformations(img):
                    data.append(x)
                    labels.append(i)

                cnt += 1
                if cnt>=30: break
                '''

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
            #data.append( cleanImage( imread(os.path.join(root, name), mode='L') ) )
            data.append( cleanImage( imread(os.path.join(root, name)) ) )
            filenames.append(name)
            #break
    
    data = np.array(data)
    return data, filenames

def get_feature_test_points_preprocessed(data_dir):
    data = []
    filenames = []
    for root, dirs, files in os.walk(data_dir):
        i = 0
        for name in files:
            data.append( preprocessImage( imread(os.path.join(root, name)) ) )
            filenames.append(name)
            #i += 1
            #if i>20: break
    
    data = np.array(data)
    return data, filenames

def saveModel(model):
    global modelName
    with open( modelName, "wb" ) as fp:
        pickle.dump( model, fp )

def loadModel():
    global modelName
    model = None
    with open( modelName, "rb" ) as fp:
        model = pickle.load(fp)
    return model

def GatherTrainTestAndEvaluate(Data_Dir):
    # Process data into feature and label arrays
    X_train, X_test, y_train, y_test = get_features_and_labels(os.path.join(Data_Dir, 'train'))

    # Train the classifier
    #clf = svm.SVC()
    #clf = svm.LinearSVC()
    clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Predict on the test set and report the metrics
    predicted = clf.predict(X_test)
    print(metrics.classification_report(y_test, predicted))

    # Save the classifier
    saveModel(clf)

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
    X_test_st2, filenames_st2 = get_feature_test_points(os.path.join(Data_Dir, 'test_stg2'))
    filenames_st2 = [ "test_stg2/" + x for x in filenames_st2 ]
    filenames = filenames + filenames_st2
    X_test = np.concatenate((X_test, X_test_st2), axis=0)
    print(X_test)
    clf = loadModel()
    #predictions = clf.predict(X_test)
    predictions = clf.predict_proba(X_test)
    #predictions = clf.decision_function(X_test) #clf.predict_proba(X_test)

    #from sklearn.preprocessing import normalize
    #predictions = normalize(1.0/( 1+np.exp(-1*predictions)), axis=1, norm='l1')
    print(predictions[0])

    writePredictionsToCsv(Data_Dir, predictions, filenames)

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

from enum import Enum
class ClassifierStage(Enum):
    Train = 1
    Test = 2
    TrainTest = 3

Data_Dir = 'C:\\Users\\t-anik\\Desktop\\personal\\KaggleData'
if __name__ == '__main__':
    Data_Dir = 'C:\\Users\\t-anik\\Desktop\\personal\\KaggleData'

    #Stage = ClassifierStage.TrainTest
    Stage = ClassifierStage.Test

    if Stage==ClassifierStage.Train or Stage==ClassifierStage.TrainTest : GatherTrainTestAndEvaluate(Data_Dir)
    if Stage==ClassifierStage.Test or Stage==ClassifierStage.TrainTest : GatherTestDataAndPredict(Data_Dir)
    #experimentOnAnImage(os.path.join(Data_Dir, 'train'))
