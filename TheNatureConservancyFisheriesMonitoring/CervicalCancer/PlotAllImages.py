import matplotlib.pylab as plt
from TestImageFlip import createImageVariations
from classifier import cleanImage
import cv2, os

def plt_st(l1,l2):
    plt.figure(figsize=(l1,l2))

if __name__ == "__main__":
    DEVELOPMENT = True
    CREATE_IMAGE_VARIATIONS = False
    data_dir = 'C:\\Users\\t-anik\\Desktop\\personal\\KaggleData\\cervical-cancer\\train'
    #classLabels = ['Type_1', 'Type_2', 'Type_3']
    classLabels = ['Type_1']
    newSize = (256,256)

    plt.figure(1)
    rows = 3
    cols = 3

    j = 0
    for i in range(len(classLabels)):
        label = classLabels[i]
        print(label)
        
        for root, dirs, files in os.walk(os.path.join(data_dir, label)):
            cnt = 0
            for name in files:
                print((os.path.join(root, name)))
                
                img = cv2.imread(os.path.join(root, name), 0)
                img = cv2.resize(img, newSize)
                print(img.shape)
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                print(img.shape)
                #try:
                #plt.imshow(img)
                j+=1
                plt.subplot(int(str(rows) + str(cols) + str(j)))
                plt.imshow(img, 'gray')

                thresh1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                    cv2.THRESH_BINARY_INV,11,2)
                #ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
                j+=1
                cnt += 1
                plt.subplot(int(str(rows) + str(cols) + str(j)))
                plt.imshow(thresh1, 'gray')

                #plt.plot(img)
                #data.append( cleanImage( img ) )
                #labels.append(i)

                if CREATE_IMAGE_VARIATIONS:
                    #img = imresize(img, newShape)
                    for x in createImageVariations(img, num_per_variation=NUMBER_PER_VARIATIONS):
                        data.append(cleanImage(x))
                        labels.append(i)
                #except:
                #    print("Got an error in cleaning the image")
                #    pass
                #break

                cnt += 1
                #if DEVELOPMENT and cnt>=1: break
                if DEVELOPMENT and cnt>=(rows*cols-1): break

    plt.show()