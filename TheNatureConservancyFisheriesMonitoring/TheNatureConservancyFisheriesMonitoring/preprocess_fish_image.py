import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from sklearn import datasets
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.image import extract_patches_2d
import classifier as NCF
import sys
import os

img = cv2.imread('C:\\Users\\t-anik\\Desktop\\personal\\KaggleData\\train\\DOL\\img_00165.jpg')

#patches = image.extract_patches_2d(img, (224, 224), max_patches=100)
#for patch in patches:
#    cv2.imshow('sadsa',patch)
#    cv2.waitKey()

label = 'DOL'
images = []
for root, dirs, files in os.walk(os.path.join(os.path.join(NCF.Data_Dir, 'train'), label)):
        cnt = 0
        print(1)
        for name in files:
            print((os.path.join(root, name)))
            img = imread(os.path.join(root, name), mode='L')
            #cv2.imshow('dst',img); cv2.waitKey();
            img = img.astype(np.float)
            #img = imread(os.path.join(root, name), mode='L')
            images.append( img )

            print(cnt)
            cnt += 1
            #if cnt>10: break
#sys.exit(1)

#faces = datasets.fetch_olivetti_faces()
print('Learning the dictionary... ')
rng = np.random.RandomState(0)
#kmeans = MiniBatchKMeans(n_clusters=81, random_state=rng, verbose=True)
kmeans = MiniBatchKMeans(n_clusters=10, random_state=rng, verbose=True)
#patch_size = (20, 20)
patch_size = (120, 120)
#display_size = (120,120,3)
display_size = (120,120)

buffer = []
index = 1
t0 = time.time()

# The online learning part: cycle over the whole dataset 6 times
print("num of images = {0}".format(len(images)))
index = 0
for _ in range(10):
    #for img in faces.images:
    for img in images:
        #cv2.imshow('dst',img.astype(np.uint8)); cv2.waitKey();
        #print(img.shape)
        data = extract_patches_2d(img, patch_size, max_patches=100,
                                  random_state=rng)
        #print(data[10].shape)
        #cv2.imshow('dst',data[10].astype(np.uint8)); cv2.waitKey();
        #print(data[10].astype(np.uint8))
        #sys.exit(1)
        data = np.reshape(data, (len(data), -1))
        buffer.append(data)
        index += 1
        if index % 10 == 0:
            data = np.concatenate(buffer, axis=0)
            data -= np.mean(data, axis=0)
            data /= np.std(data, axis=0)
            kmeans.partial_fit(data)
            buffer = []
        if index % 100 == 0:
            print('Partial fit of %4i out of %i'
                  % (index, 10 * len(images)))

dt = time.time() - t0
print('done in %.2fs.' % dt)

#plt.figure(figsize=(4.2, 4))
plt.figure(figsize=(60, 60))
for i, patch in enumerate(kmeans.cluster_centers_):
    plt.subplot(9, 9, i + 1)
    #cv2.imshow('dst',patch.reshape(display_size).astype(np.uint8)) ; cv2.waitKey();
    plt.imshow(patch.reshape(display_size), cmap=plt.cm.gray,
               interpolation='nearest')
    #cv2.imshow('dst',patch.reshape(display_size )); cv2.waitKey();
    plt.xticks(())
    plt.yticks(())


#plt.suptitle('Patches of faces\nTrain time %.1fs on %d patches' %
#             (dt, 8 * len(faces.images)), fontsize=16)
plt.suptitle('Patches of faces\nTrain time %.1fs on %d patches' %
             (dt, 8 * len(images)), fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

plt.show()