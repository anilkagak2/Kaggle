from featuresFromCNN import *
from sklearn import svm 
import numpy as np
import sys
import cv2

bin_n = 16
def hog(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    '''from skimage.feature import hog
    from skimage import data, color, exposure

    fd = hog(img, orientations=16, pixels_per_cell=(4, 4), cells_per_block=(1, 1), visualise=False)
    return fd
'''
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist

def create_hog_features(X):
    N = len(X)
    num_features = bin_n*4#50176 #64
    F = np.zeros((N,num_features))
    for i in range(N):
        F[i] = hog(X[i])[:num_features]
    return F

if __name__ == "__main__":
    #Put here the number of your experiment
    EXPERIMENT_NUMBER = '2017-05-01'  

    #Put here the path where you downloaded all kaggle data
    DATA_PATH = 'C:\\Users\\t-anik\\Desktop\\personal\\KaggleData\\cervical-cancer\\'
    TRAIN_PATH = DATA_PATH + 'train\\'
    TEST_PATH = DATA_PATH + 'test\\'

    # Path and variables
    SUBMIT_OUTPUT = DATA_PATH + 'submit_hog_features-' + EXPERIMENT_NUMBER + '.csv'
    TRAIN_TEST_DATA_FILE = DATA_PATH + "train_test_data_file.npz"

    num_classes = len(classLabels)
    LOAD_FROM_DISK = False
    img_rows, img_cols, img_channels = 224, 224, 3

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

    n_train, n_val, n_test = len(X_train), len(X_val), len(X_test)
    print("n_train={0}, n_val={1} and n_test={2}".format(n_train, n_val, n_test))
    
    X_train = X_train.reshape((n_train, img_rows, img_cols, img_channels))
    X_val = X_val.reshape((n_val, img_rows, img_cols, img_channels))
    X_test = X_test.reshape((n_test, img_rows, img_cols, img_channels))

    from sklearn import preprocessing
    X_train = preprocessing.normalize( X_train.reshape(n_train, -1) )
    X_val = preprocessing.normalize( X_val.reshape(n_val, -1) )
    X_test = preprocessing.normalize( X_test.reshape(n_test, -1) )

    #X_train = create_hog_features(X_train)
    #X_val = create_hog_features(X_val)
    #X_test = create_hog_features(X_test)

    from sklearn.ensemble import RandomForestClassifier
    n_estimators = 4000
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
    #clf = svm.SVC(probability=True)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_val)
    print(predicted[0])
    print(metrics.classification_report(y_val, predicted))
    predicted_prob = clf.predict_proba(X_val)
    print("log-loss = {0:.3f}".format(log_loss(y_val, predicted_prob)))

    predictions = clf.predict_proba(X_test)
    print(predictions[0])

    writePredictionsToCsv(predictions, filenames, SUBMIT_OUTPUT)