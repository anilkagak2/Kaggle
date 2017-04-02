from __future__ import print_function
import classifier as NCF
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import sys
import os

batch_size = 128
num_classes = len(NCF.classLabels)
epochs = 12

img_rows, img_cols = 64, 64
x_train, x_test, x_valid, y_train, y_test, y_valid = NCF.get_images_labels(os.path.join(NCF.Data_Dir, 'train'))

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
x_valid = x_valid.reshape(x_valid.shape[0], img_rows, img_cols, 3)
input_shape = (img_rows, img_cols, 3)

x_train /= 255
x_test /= 255
x_valid /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

print(y_train[:10])

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

print(y_train[:10])

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_valid, y_valid))
score = model.evaluate(x_test, y_test, verbose=0)
print(score)
print("Test loss : ", score[0])
print("Test accuracy : ", score[1])

x_test, filenames = NCF.get_feature_test_points_preprocessed(os.path.join(NCF.Data_Dir, 'test_stg1'))
x_test /= 255
predictions = model.predict(x_test, batch_size=64)
print(predictions[:2])
NCF.writePredictionsToCsv(NCF.Data_Dir, predictions, filenames)