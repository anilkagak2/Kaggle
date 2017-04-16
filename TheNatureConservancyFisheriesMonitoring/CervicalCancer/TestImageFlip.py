import keras
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

img = cv2.imread('messi5.jpg')
print(img.shape)

cv2.imshow('dst_rt', img)
cv2.waitKey(0)

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=True,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

x_train = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
y_train = np.array([1])
datagen.fit(x_train)

for X_batch, Y_batch in datagen.flow(x_train, y_train, batch_size=1):
    img = X_batch[0]
    img = img.astype(np.uint8)
    cv2.imshow('dst_rt', img)
    cv2.waitKey(0)
    