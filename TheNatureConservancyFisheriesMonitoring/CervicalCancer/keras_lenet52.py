#from classifier import *
from featuresFromCNN import *
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import sys

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
#X_train = preprocessing.normalize( X_train.reshape(n_train, -1) )
#X_val = preprocessing.normalize( X_val.reshape(n_val, -1) )
#X_test = preprocessing.normalize( X_test.reshape(n_test, -1) )

batch_size = 16
num_classes = len(classLabels)
epochs = 2000
initial_epoch = 1000
data_augmentation = True
load_model_weights = True
previous_mode_file = "weights.999-0.8077.hdf5"

# Convert class vectors to binary class matrices.
old_y_train, old_y_val = y_train, y_val
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same',
                 input_shape=X_train.shape[1:], kernel_initializer='glorot_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (5, 5), kernel_initializer='glorot_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='glorot_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (5, 5), kernel_initializer='glorot_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, kernel_initializer='glorot_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, kernel_initializer='glorot_uniform'))
model.add(BatchNormalization())
model.add(Activation('softmax'))
model.summary()

# initiate RMSprop optimizer
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
opt = keras.optimizers.Adadelta(lr=0.1, decay=1e-3)
#opt = keras.optimizers.Adadelta()
opt = keras.optimizers.Adam(lr=0.0001, decay=1e-6)
#opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-5)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy', 'categorical_crossentropy'])

filepath = os.path.join( os.path.join(DATA_PATH, "keras_lenet52"), "weights.{epoch:02d}-{val_loss:.4f}.hdf5" )
if not data_augmentation:
    print('Not using data augmentation.')
    if load_model_weights: model.load_weights( os.path.join( os.path.join(DATA_PATH, 'keras_lenet52\\back\\'), previous_mode_file) )
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_val, y_val),
              shuffle=True,
              callbacks=[keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=50)])
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)

    if load_model_weights: model.load_weights( os.path.join( os.path.join(DATA_PATH, 'keras_lenet52\\back\\'), previous_mode_file) )
    steps_per_epoch = X_train.shape[0] // batch_size #x_train.shape[0]#
    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(X_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        validation_data=(X_val, y_val),
                        callbacks=[keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=50)],
                        initial_epoch = initial_epoch,
                        )#callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=1, mode='auto')])

predicted_prob = model.predict(X_val)
from sklearn import metrics
predicted = np.argmax(predicted_prob, axis=1) 
print(metrics.classification_report(old_y_val, predicted))
print("log-loss = {0:.3f}".format(metrics.log_loss(old_y_val, predicted_prob)))

predictionsFilename = "newpredictions-4conv-1fc_" + str(img_rows) + "_x_" + str(img_cols) + ".csv"
predictions = model.predict(X_test)
print(predictions[0])
writePredictionsToCsv(predictions, filenames, SUBMIT_OUTPUT)