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
from TestImageFlip import createImageVariations

DEVELOPMENT = False
LOAD_FROM_DISK = True
img_rows, img_cols, nchannels = 128, 128, 3
#img_rows, img_cols, nchannels = 60, 40, 3

#img_rows, img_cols, nchannels = 60, 40, 3
newShape = (img_rows, img_cols)
modelName = "model-svc-default.bin"
predictionsFilename = "predictions-RandomForestClassifier_" + str(img_rows) + "_x_" + str(img_cols) + ".csv"
trainTestFilename = 'train_test_data_with_augmentation.npz'
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
                    img = imresize(img, newShape)
                    data.append( cleanImage( img ) )
                    labels.append(i)

                    for x in createImageVariations(img, num_per_variation=2):
                        data.append(cleanImage(x))
                        labels.append(i)
                except:
                    print("Got an error in cleaning the image")
                    pass
                #break

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

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def GatherTrainTestAndEvaluate(Data_Dir):
    # Process data into feature and label arrays
    train_test_data_file = os.path.join(Data_Dir, trainTestFilename)
    if LOAD_FROM_DISK:
        X_train, X_test, y_train, y_test = get_features_and_labels(os.path.join(Data_Dir, 'train'))
        #np.savez(train_test_data_file, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    else:
        npzfile = np.load(train_test_data_file)
        X_train, X_test, y_train, y_test = npzfile['X_train'], npzfile['X_test'], npzfile['y_train'], npzfile['y_test']

    # Train the classifier
    new_X_train, new_y_train = X_train, y_train
    new_X_test, new_y_test = X_test, y_test

    print(new_X_train.shape)
    from time import time
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from scipy.stats import randint as sp_randint
    from sklearn.metrics import log_loss

    TUNE = False
    if TUNE:
        print("Will be doing parameter tuning...")
        clf = RandomForestClassifier(n_estimators=10)

        # specify parameters and distributions to sample from
        param_dist = {"max_depth": [3, None],
                      "max_features": sp_randint(1, 11),
                      "min_samples_split": sp_randint(2, 11),
                      "min_samples_leaf": sp_randint(1, 11),
                      "bootstrap": [True, False],
                      "criterion": ["gini", "entropy"]}

        # run randomized search
        n_iter_search = 20
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                           n_iter=n_iter_search, n_jobs=-1,
                                           class_weight='balanced')

        start = time()
        random_search.fit(new_X_train, new_y_train)
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
              " parameter settings." % ((time() - start), n_iter_search))
        report(random_search.cv_results_)
        sys.exit(1)

    from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
    #scaler = MinMaxScaler()
    #scaler.fit(new_X_train)
    #new_X_train, new_X_test = scaler.transform(new_X_train), scaler.transform(new_X_test)
    clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, bootstrap=False, 
                                 class_weight=None)
    clf.fit(new_X_train, new_y_train)
    predicted = clf.predict(new_X_test)
    print(metrics.classification_report(new_y_test, predicted))
    predicted_prob = clf.predict_proba(new_X_test)
    print("log-loss = {0:.3f}".format(log_loss(new_y_test, predicted_prob)))

    # Save the classifier
    '''saveModel(Data_Dir, clf)

    if LOAD_FROM_DISK:
        print("will be saving to disk now.. not sure if it'll be done")
        np.savez(train_test_data_file, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)'''
    return clf

def writePredictionsToCsv(Data_Dir, predictions, filenames):
    global classLabels
    global predictionsFilename
    import csv
    with open( os.path.join(Data_Dir, predictionsFilename), 'w') as fp:
        writer = csv.writer(fp, lineterminator='\n')
        writer.writerow(["image_name"] + classLabels)
        for i in range(len(filenames)):
            writer.writerow([filenames[i]] + [ str(x) for x in predictions[i]])

def GatherTestDataAndPredict(Data_Dir, clf=None):
    #print("Prediction..")
    X_test, filenames = get_feature_test_points(os.path.join(Data_Dir, 'test'))
    '''X_test_st2, filenames_st2 = get_feature_test_points(os.path.join(Data_Dir, 'test_stg2'))
    filenames_st2 = [ "test_stg2/" + x for x in filenames_st2 ]
    filenames = filenames + filenames_st2
    X_test = np.concatenate((X_test, X_test_st2), axis=0)'''
    #print(X_test)
    if clf is None:
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
    #Stage = ClassifierStage.Train
    #Stage = ClassifierStage.Test
    clf = None
    if Stage==ClassifierStage.Train or Stage==ClassifierStage.TrainTest : clf=GatherTrainTestAndEvaluate(Data_Dir)
    if Stage==ClassifierStage.Test or Stage==ClassifierStage.TrainTest : GatherTestDataAndPredict(Data_Dir, clf)
    #experimentOnAnImage(os.path.join(Data_Dir, 'train'))
