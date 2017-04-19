import keras
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, random_rotation, random_shift, random_zoom, random_shear

def createHistogramNormalizedImage(img):
    SAVE_DIR = True
    import cv2
    variations = [img]
    print(np.max(img))

    img = cv2.equalizeHist(img)
    print(np.max(img))
    variations.append(img)
    
    if SAVE_DIR:
        i=0
        for img in variations:
            cv2.imwrite(output_dir + str(i) + ".png", img)
            i += 6
        #    cv2.imshow('dst_rt', img)
        #    cv2.waitKey(0)

    #print(len(variations))
    return variations

def createImageVariations(img, num_per_variation=2):
    SAVE_DIR = False
    import cv2
    zx,zy = 1,2
    rotation_rng = 30
    sx, sy= 0.1, 0.1,
    shear_intensity = 0.5
    variations = []
    for _ in range(num_per_variation):
        variations.append( random_rotation( img, rotation_rng, row_axis=0, col_axis=1, channel_axis=2 ) )
        variations.append( random_shift( img, sx, sy, row_axis=0, col_axis=1, channel_axis=2 ) )
        variations.append( random_zoom( img, (zx,zy), row_axis=0, col_axis=1, channel_axis=2 ) )
        variations.append( random_zoom( img, (zy,zx), row_axis=0, col_axis=1, channel_axis=2 ) )
        variations.append( random_shear( img, shear_intensity, row_axis=0, col_axis=1, channel_axis=2 ) )

    variations.append( cv2.flip(img, 1) )
    variations.append( cv2.flip(img, 0) )
    if SAVE_DIR:
        i=0
        for img in variations:
            cv2.imwrite(output_dir + str(i) + ".png", img)
            i += 6
        #    cv2.imshow('dst_rt', img)
        #    cv2.waitKey(0)

    #print(len(variations))
    return variations

def imageGeneratorExample():
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

if __name__ == "__main__":
    image_file = "messi5.jpg"
    image_file = "C:\\Users\\t-anik\\Desktop\\personal\\KaggleData\cervical-cancer\\train\\Type_1\\1389.jpg"
    output_dir = "C:\\Users\\t-anik\\Desktop\\personal\\KaggleData\cervical-cancer\\samples\\"

    #img = cv2.imread('messi5.jpg')
    #img = cv2.imread(image_file, 0)
    img = cv2.imread(image_file)
    print(img.shape)
    print(img.dtype)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    print(img.shape)
    print(img.dtype)

    #cv2.imshow('dst_rt', img)
    #cv2.waitKey(0)

    createHistogramNormalizedImage(img)

    #createImageVariations(img)