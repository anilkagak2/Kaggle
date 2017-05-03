import h5py
from keras.models import Model
from keras.layers import Input, Activation, Concatenate
from keras.layers import Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import AveragePooling2D
from featuresFromCNN import *

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import tensorflow as tf
import sys

def SqueezeNet(nb_classes, inputs=(3, 224, 224)):
    """ Keras Implementation of SqueezeNet(arXiv 1602.07360)

    @param nb_classes: total number of final categories

    Arguments:
    inputs -- shape of the input images (channel, cols, rows)

    """
    print(nb_classes)
    input_img = Input(shape=inputs)
    conv1 = Convolution2D(
        96, (7, 7), activation='relu', kernel_initializer='glorot_uniform',
        strides=(2, 2), padding='same', name='conv1',
        data_format="channels_first")(input_img)
    maxpool1 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool1',
        data_format="channels_first")(conv1)
    fire2_squeeze = Convolution2D(
        16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_squeeze',
        data_format="channels_first")(maxpool1)
    fire2_expand1 = Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_expand1',
        data_format="channels_first")(fire2_squeeze)
    fire2_expand2 = Convolution2D(
        64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_expand2',
        data_format="channels_first")(fire2_squeeze)
    merge2 = Concatenate(axis=1)([fire2_expand1, fire2_expand2])

    fire3_squeeze = Convolution2D(
        16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_squeeze',
        data_format="channels_first")(merge2)
    fire3_expand1 = Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_expand1',
        data_format="channels_first")(fire3_squeeze)
    fire3_expand2 = Convolution2D(
        64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_expand2',
        data_format="channels_first")(fire3_squeeze)
    merge3 = Concatenate(axis=1)([fire3_expand1, fire3_expand2])

    fire4_squeeze = Convolution2D(
        32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_squeeze',
        data_format="channels_first")(merge3)
    fire4_expand1 = Convolution2D(
        128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_expand1',
        data_format="channels_first")(fire4_squeeze)
    fire4_expand2 = Convolution2D(
        128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_expand2',
        data_format="channels_first")(fire4_squeeze)
    merge4 = Concatenate(axis=1)([fire4_expand1, fire4_expand2])
    maxpool4 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool4',
        data_format="channels_first")(merge4)

    fire5_squeeze = Convolution2D(
        32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_squeeze',
        data_format="channels_first")(maxpool4)
    fire5_expand1 = Convolution2D(
        128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_expand1',
        data_format="channels_first")(fire5_squeeze)
    fire5_expand2 = Convolution2D(
        128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_expand2',
        data_format="channels_first")(fire5_squeeze)
    merge5 = Concatenate(axis=1)([fire5_expand1, fire5_expand2])

    fire6_squeeze = Convolution2D(
        48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_squeeze',
        data_format="channels_first")(merge5)
    fire6_expand1 = Convolution2D(
        192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_expand1',
        data_format="channels_first")(fire6_squeeze)
    fire6_expand2 = Convolution2D(
        192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_expand2',
        data_format="channels_first")(fire6_squeeze)
    merge6 = Concatenate(axis=1)([fire6_expand1, fire6_expand2])

    fire7_squeeze = Convolution2D(
        48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire7_squeeze',
        data_format="channels_first")(merge6)
    fire7_expand1 = Convolution2D(
        192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire7_expand1',
        data_format="channels_first")(fire7_squeeze)
    fire7_expand2 = Convolution2D(
        192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire7_expand2',
        data_format="channels_first")(fire7_squeeze)
    merge7 = Concatenate(axis=1)([fire7_expand1, fire7_expand2])

    fire8_squeeze = Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire8_squeeze',
        data_format="channels_first")(merge7)
    fire8_expand1 = Convolution2D(
        256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire8_expand1',
        data_format="channels_first")(fire8_squeeze)
    fire8_expand2 = Convolution2D(
        256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire8_expand2',
        data_format="channels_first")(fire8_squeeze)
    merge8 = Concatenate(axis=1)([fire8_expand1, fire8_expand2])

    maxpool8 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool8',
        data_format="channels_first")(merge8)
    fire9_squeeze = Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire9_squeeze',
        data_format="channels_first")(maxpool8)
    fire9_expand1 = Convolution2D(
        256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire9_expand1',
        data_format="channels_first")(fire9_squeeze)
    fire9_expand2 = Convolution2D(
        256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire9_expand2',
        data_format="channels_first")(fire9_squeeze)
    merge9 = Concatenate(axis=1)([fire9_expand1, fire9_expand2])

    fire9_dropout = Dropout(0.5, name='fire9_dropout')(merge9)
    conv10 = Convolution2D(
        nb_classes, (1, 1), kernel_initializer='glorot_uniform',
        padding='valid', name='conv10',
        data_format="channels_first")(fire9_dropout)
    # The size should match the output of conv10
    avgpool10 = AveragePooling2D(
        (13, 13), name='avgpool10',
        data_format="channels_first")(conv10)

    flatten = Flatten(name='flatten')(avgpool10)
    softmax = Activation("softmax", name='softmax')(flatten)

    return Model(inputs=input_img, outputs=softmax)

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
    TEST_PATH = DATA_PATH + 'test\\'

    # Path and variables
    SUBMIT_OUTPUT = DATA_PATH + 'submit_cnn_features-' + EXPERIMENT_NUMBER + '.csv'
    TRAIN_TEST_DATA_FILE = DATA_PATH + "train_test_data_file.npz"
    filepath = os.path.join( os.path.join(DATA_PATH, "SqueezeNet"), "weights.{epoch:02d}-{val_loss:.4f}.hdf5" )

    batch_size = 32
    num_classes = len(classLabels)
    epochs = 5000
    data_augmentation = True
    load_model_weights = True
    previous_mode_file = "weights.499-0.9307.hdf5"

    LOAD_FROM_DISK = False
    load_model_weights = True
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
        #X_train, X_val, X_test = X_train.astype(np.float32), X_val.astype(np.float32), X_test.astype(np.float32)
        #np.savez(TRAIN_TEST_DATA_FILE, X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train, y_val=y_val, filenames=filenames)
        #sys.exit(1)

    n_train, n_val, n_test = len(X_train), len(X_val), len(X_test)
    print("n_train={0}, n_val={1} and n_test={2}".format(n_train, n_val, n_test))
    
    X_train = X_train.reshape((n_train, img_channels, img_rows, img_cols))
    X_val = X_val.reshape((n_val, img_channels, img_rows, img_cols))
    X_test = X_test.reshape((n_test, img_channels, img_rows, img_cols))

    # Convert class vectors to binary class matrices.
    old_y_train, old_y_val = y_train, y_val
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)

    opt = keras.optimizers.Adadelta(lr=0.1, decay=1e-3)
    model = SqueezeNet(len(classLabels))
    print(model.summary())

    model.compile(
            optimizer=opt, loss='categorical_crossentropy',
            metrics=['accuracy'])
    if load_model_weights: model.load_weights( os.path.join( os.path.join(DATA_PATH, "SqueezeNet"), previous_mode_file) )
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_val, y_val),
              shuffle=True,
              initial_epoch = 700,
              callbacks=[keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=500)])

    predicted_prob = model.predict(X_val)
    from sklearn import metrics
    predicted = np.argmax(predicted_prob, axis=1) 
    print(metrics.classification_report(old_y_val, predicted))
    print("log-loss = {0:.3f}".format(metrics.log_loss(old_y_val, predicted_prob)))
    predictions = model.predict(X_test)
    print(predictions[0])

    writePredictionsToCsv(predictions, filenames, SUBMIT_OUTPUT)