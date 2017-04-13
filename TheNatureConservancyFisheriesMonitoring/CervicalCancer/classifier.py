import numpy as np
import pickle
import keras
from sklearn.preprocessing import normalize
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from scipy.ndimage import imread
from scipy.misc import imresize
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
import cv2
import sys

DEVELOPMENT = False
LOAD_FROM_DISK = False

img_rows, img_cols = 60, 40
newShape = (img_rows, img_cols)
modelName = "model-svc-default.bin"
predictionsFilename = "predictions-RandomForestClassifier_" + str(img_rows) + "_x_" + str(img_cols) + ".csv"
classLabels = ['Type_1', 'Type_2', 'Type_3']

def cleanImage(im):
    global newShape
    im = imresize(im, newShape)
    im = im.astype(np.float)
    im = im/ 256.0
    im = im.flatten()
    return im

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
                try:
                    img = cleanImage( img )
                    data.append( img )
                    labels.append(i)
                except:
                    print("Got an error in cleaning the image")
                    pass
                #break

                '''
                for x in getImageTransformations(img):
                    data.append(x)
                    labels.append(i)
                '''

                cnt += 1
                if DEVELOPMENT and cnt>=30: break

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
        cnt = 0
        for name in files:
            #print((os.path.join(root, name)))
            #data.append( cleanImage( imread(os.path.join(root, name), mode='L') ) )
            data.append( cleanImage( imread(os.path.join(root, name)) ) )
            filenames.append(name)
            cnt += 1
            if DEVELOPMENT and cnt>=30: break
    
    data = np.array(data)
    return data, filenames

def saveModel(Data_Dir, model):
    global modelName
    with open( os.path.join(Data_Dir, modelName), "wb" ) as fp:
        pickle.dump( model, fp )

def loadModel(Data_Dir):
    global modelName
    model = None
    with open( os.path.join(Data_Dir, modelName), "rb" ) as fp:
        model = pickle.load(fp)
    return model

def GatherTrainTestAndEvaluate(Data_Dir):
    # Process data into feature and label arrays
    train_test_data_file = os.path.join(Data_Dir, 'train_test_data.npz')
    if LOAD_FROM_DISK:
        X_train, X_test, y_train, y_test = get_features_and_labels(os.path.join(Data_Dir, 'train'))
        np.savez(train_test_data_file, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    else:
        npzfile = np.load(train_test_data_file)
        X_train, X_test, y_train, y_test = npzfile['X_train'], npzfile['X_test'], npzfile['y_train'], npzfile['y_test']

    # Train the classifier
    #clf = svm.SVC()
    #clf = svm.LinearSVC()
    '''total_trees = 500
    clf = RandomForestClassifier(n_estimators=total_trees, class_weight='balanced', n_jobs=-1)
    clf.fit(X_train, y_train)

    predicted = clf.predict(X_test)
    print(metrics.classification_report(y_test, predicted))'''

    datagen = ImageDataGenerator(
        samplewise_center=True,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')

    new_X_train, new_y_train = X_train.copy(), y_train.copy()
    new_X_test, new_y_test = X_test.copy(), y_test.copy()
    epochs = 20
    # here's a more "manual" example
    for e in range(epochs):
        print('Epoch = {0}'.format(e))
        batches = 0
        batch_size = 4096
        k_x_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
        for X_batch, Y_batch in datagen.flow(k_x_train, y_train, batch_size=batch_size):
            X_batch = X_batch.reshape(X_batch.shape[0], img_rows*img_cols*3)
            print(X_batch.shape)
            print(new_X_train.shape)
            new_X_train = np.concatenate((new_X_train, X_batch))
            new_y_train = np.concatenate((new_y_train, Y_batch))
            print("batch = {0}".format(batches))
            batches += 1
            if batches >= len(X_train) / (2*1024): break

        k_x_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
        for X_batch, Y_batch in datagen.flow(k_x_test, y_test, batch_size=batch_size):
            X_batch = X_batch.reshape(X_batch.shape[0], img_rows*img_cols*3)
            print(X_batch.shape)
            print(new_X_train.shape)
            new_X_test = np.concatenate((new_X_test, X_batch))
            new_y_test = np.concatenate((new_y_test, Y_batch))
            print("batch = {0}".format(batches))
            batches += 1
            if batches >= len(X_test) / (2*1024): break

    clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
    clf.fit(new_X_train, new_y_train)
    predicted = clf.predict(new_X_test)
    print(metrics.classification_report(new_y_test, predicted))

    # Save the classifier
    saveModel(Data_Dir, clf)

def writePredictionsToCsv(Data_Dir, predictions, filenames):
    global classLabels
    global predictionsFilename
    import csv
    with open( os.path.join(Data_Dir, predictionsFilename), 'w') as fp:
        writer = csv.writer(fp, lineterminator='\n')
        writer.writerow(["image_name"] + classLabels)
        for i in range(len(filenames)):
            writer.writerow([filenames[i]] + [ str(x) for x in predictions[i]])

def GatherTestDataAndPredict(Data_Dir):
    print("Prediction..")
    X_test, filenames = get_feature_test_points(os.path.join(Data_Dir, 'test'))
    '''X_test_st2, filenames_st2 = get_feature_test_points(os.path.join(Data_Dir, 'test_stg2'))
    filenames_st2 = [ "test_stg2/" + x for x in filenames_st2 ]
    filenames = filenames + filenames_st2
    X_test = np.concatenate((X_test, X_test_st2), axis=0)'''
    print(X_test)
    clf = loadModel(Data_Dir)
    #predictions = clf.predict(X_test)
    predictions = clf.predict_proba(X_test)
    print(predictions[0])

    writePredictionsToCsv(Data_Dir, predictions, filenames)

from enum import Enum
class ClassifierStage(Enum):
    Train = 1
    Test = 2
    TrainTest = 3

Data_Dir = 'C:\\Users\\t-anik\\Desktop\\personal\\KaggleData\\cervical-cancer'
if __name__ == '__main__':
    Data_Dir = 'C:\\Users\\t-anik\\Desktop\\personal\\KaggleData\\cervical-cancer'

    Stage = ClassifierStage.TrainTest
    #Stage = ClassifierStage.Test

    if Stage==ClassifierStage.Train or Stage==ClassifierStage.TrainTest : GatherTrainTestAndEvaluate(Data_Dir)
    if Stage==ClassifierStage.Test or Stage==ClassifierStage.TrainTest : GatherTestDataAndPredict(Data_Dir)
    #experimentOnAnImage(os.path.join(Data_Dir, 'train'))
