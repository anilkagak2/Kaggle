from classifier import *
import sys
import numpy as np
import cv2
import sklearn.metrics as sklm

from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.datasets import mnist
from keras import backend as K

#Get back the convolutional part of a VGG network trained on ImageNet
#model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
#model_vgg16_conv.summary()

img_dim_ordering = 'tf'
K.set_image_dim_ordering(img_dim_ordering)

# the model
def pretrained_model(img_shape, num_classes, layer_type):
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
    #model_vgg16_conv.summary()
    
    #Create your own input format
    keras_input = Input(shape=img_shape, name = 'image_input')
    
    #Use the generated model 
    output_vgg16_conv = model_vgg16_conv(keras_input)
    
    #Add the fully-connected layers 
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation=layer_type, name='fc1')(x)
    x = Dense(4096, activation=layer_type, name='fc2')(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    #Create your own model 
    pretrained_model = Model(inputs=keras_input, outputs=x)
    pretrained_model.summary()
    pretrained_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return pretrained_model

train_test_data_file = os.path.join(Data_Dir, 'train_test_data.npz')
if LOAD_FROM_DISK:
    X_train, X_test, y_train, y_test = get_features_and_labels(os.path.join(Data_Dir, 'train'))
    np.savez(train_test_data_file, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
else:
    npzfile = np.load(train_test_data_file)
    X_train, X_test, y_train, y_test = npzfile['X_train'], npzfile['X_test'], npzfile['y_train'], npzfile['y_test']

samplewise_center=True
rotation_range=90
width_shift_range=0
height_shift_range=0
shear_range=0
zoom_range=0
horizontal_flip=True
fill_mode='nearest'

datagen = ImageDataGenerator(
        samplewise_center=samplewise_center,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        fill_mode=fill_mode)

valid_datagen = ImageDataGenerator(
        samplewise_center=samplewise_center,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        fill_mode=fill_mode)

X_train = X_train.reshape((len(X_train), img_rows, img_cols, nchannels))
X_test = X_test.reshape((len(X_test), img_rows, img_cols, nchannels))

# training the model
model = pretrained_model(X_train.shape[1:], len(set(y_train)), 'relu')
batch_size = 32
epochs = 10
steps_per_epoch = len(X_train) // 64
model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),\
    steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1,\
    validation_data=valid_datagen.flow(X_test, y_test, batch_size=batch_size),\
    validation_steps=len(X_test))

#model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test))

predictions = model.predict(X_test, batch_size=32)
print(predictions[:2])

from sklearn import metrics
predicted = np.argmax(predictions, axis=1)
print(metrics.classification_report(y_test, predicted))