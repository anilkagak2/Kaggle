import sys,os
import numpy as np
import glob
from matplotlib import pyplot as plt
import cv2
import time
from scipy.ndimage import imread
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import cntk
from cntk import load_model
from cntk.ops import combine
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs
from cervix_segmentation import getCleanedImageFromRGBImage
#from lightgbm.sklearn import LGBMRegressor
from cntk.device import set_default_device, gpu
set_default_device(gpu(0))

DEVELOPMENT = False
LOAD_FROM_DISK = True
classLabels = ['Type_1', 'Type_2', 'Type_3']

def cleanImage(im):
    #cv2.imshow('Color input image', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

    #im = 255.0 / np.amax(im) * im
    im = getCleanedImageFromRGBImage(im)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2YCR_CB)
    im[:,:,0] = cv2.equalizeHist(im[:,:,0])
    im = cv2.cvtColor(im, cv2.COLOR_YCR_CB2RGB)
    im = cv2.resize(im, (224, 224))
    im = im.astype(np.float)
    im = im.reshape((3,224,224))

    #cv2.imshow('Histogram equalized', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    #cv2.waitKey(0)

    return im

def get_extractor():
    node_name = "z.x"
    loaded_model  = load_model(MODEL_PATH)
    node_in_graph = loaded_model.find_by_name(node_name)
    output_nodes  = combine([node_in_graph.owner])
    return output_nodes

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
                try:
                    img = imread(os.path.join(root, name))
                    data.append( cleanImage( img ) )
                    labels.append(i)
                except:
                    print("Got an error in cleaning the image")
                    pass

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
            data.append( cleanImage( imread(os.path.join(root, name)) ) )
            filenames.append(name)
            cnt += 1
            if DEVELOPMENT and cnt>=30: break
    
    data = np.array(data)
    return data, filenames

def writePredictionsToCsv(predictions, filenames, predictionsFilename):
    global classLabels
    import csv
    with open( predictionsFilename, 'w') as fp:
        writer = csv.writer(fp, lineterminator='\n')
        writer.writerow(["image_name"] + classLabels)
        for i in range(len(filenames)):
            writer.writerow([filenames[i]] + [ str(x) for x in predictions[i]])

def evaluate_network(net, x_test, n_test, n_features, batch_size=128):
    x_features = np.zeros((n_test, n_features), dtype=np.float32)
    n_batches = n_test // batch_size
    if n_test % batch_size != 0: n_batches += 1
    for i in range(n_batches):
        xbatch = x_test[i*batch_size:(i+1)*batch_size]
        n_xbatch = len(xbatch)
        #print("n_xbatch = {0}".format(n_xbatch))
        x_features[i*batch_size:i*batch_size+n_xbatch] = net.eval(xbatch)[0].reshape((n_xbatch, n_features))
    return x_features

if __name__ == "__main__":
    print("System version:", sys.version, "\n")
    print("CNTK version:",cntk.__version__)

    #Put here the number of your experiment
    EXPERIMENT_NUMBER = '2017-04-25' 

    #Put here the path to the downloaded ResNet model
    MODEL_PATH = 'C:\\Users\\t-anik\\Desktop\\personal\\KaggleData\\ResNet_152.model' 

    #Put here the path where you downloaded all kaggle data
    DATA_PATH = 'C:\\Users\\t-anik\\Desktop\\personal\\KaggleData\\cervical-cancer\\'
    TRAIN_PATH = DATA_PATH + 'train\\'
    #TRAIN_PATH = DATA_PATH + 'nodupe\\train\\'
    TEST_PATH = DATA_PATH + 'test\\'

    # Path and variables
    SUBMIT_OUTPUT = DATA_PATH + 'submit_cnn_features-' + EXPERIMENT_NUMBER + '.csv'
    TRAIN_TEST_DATA_FILE = DATA_PATH + "train_test_data_file.npz"

    if LOAD_FROM_DISK:
        X_train, X_val, y_train, y_val = get_features_and_labels(TRAIN_PATH)
        X_test, filenames = get_feature_test_points(TEST_PATH)
        X_train, X_val, X_test = X_train.astype(np.float32), X_val.astype(np.float32), X_test.astype(np.float32)
        np.savez(TRAIN_TEST_DATA_FILE, X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train, y_val=y_val, filenames=filenames)
        sys.exit(1)
    else:
        npzfile = np.load(TRAIN_TEST_DATA_FILE)
        X_train, X_val, y_train, y_val = npzfile['X_train'], npzfile['X_val'], npzfile['y_train'], npzfile['y_val']
        X_test, filenames =  npzfile['X_test'],  npzfile['filenames']
        #X_train, X_val, X_test = X_train.astype(np.float32), X_val.astype(np.float32), X_test.astype(np.float32)
        #np.savez(TRAIN_TEST_DATA_FILE, X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train, y_val=y_val, filenames=filenames)
        #sys.exit(1)

    net = get_extractor()
    n_train, n_val, n_test = len(X_train), len(X_val), len(X_test)
    print("n_train={0}, n_val={1} and n_test={2}".format(n_train, n_val, n_test))
    #n_features = net.eval(X_train[0])[0].shape[1]
    #print("n_features = {0}".format(n_features))

    #X_train = evaluate_network(net, X_train, n_train, n_features, batch_size=128)
    #X_val = evaluate_network(net, X_val, n_val, n_features, batch_size=128)
    #X_test = evaluate_network(net, X_test, n_test, n_features, batch_size=128)
    X_train = X_train.reshape((n_train, -1))
    X_val = X_val.reshape((n_val, -1))
    X_test = X_test.reshape((n_test, -1))

    '''
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    clf = XGBClassifier(n_estimators=1000, nthread=-1)
    #clf = LGBMClassifier(n_estimators=1, nthread=-1)
    #clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True, eval_metric='l2', early_stopping_rounds=300)
    #clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True, eval_metric='logloss', early_stopping_rounds=300)
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True, eval_metric='mlogloss', early_stopping_rounds=300)

    '''
    n_estimators = 1000
    n_jobs = -1
    bootstrap = False
    class_weight = None
    criterion = 'gini'
    max_depth = None
    min_samples_split = 2#8
    min_samples_leaf = 1#4#
    max_features = 'auto'#9#

    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs, bootstrap=bootstrap, 
                                 class_weight=class_weight, criterion=criterion, max_depth=max_depth,
                                 min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                 max_features=max_features)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_val)
    print(predicted[0])
    print(metrics.classification_report(y_val, predicted))
    predicted_prob = clf.predict_proba(X_val)
    print("log-loss = {0:.3f}".format(log_loss(y_val, predicted_prob)))

    predictions = clf.predict_proba(X_test)
    print(predictions[0])

    writePredictionsToCsv(predictions, filenames, SUBMIT_OUTPUT)