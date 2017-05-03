import sys,os
import numpy as np
import cv2
from scipy.ndimage import imread
from sklearn.model_selection import train_test_split
from cervix_segmentation import getCleanedImageFromRGBImage

DEVELOPMENT = False
CNTK = False
classLabels = ['Type_1', 'Type_2', 'Type_3']

rows, cols = 224, 224

def get_features_and_labels(data_dir, out_data_dir):
    global classLabels
    labels = []
    data = []
    for i in range(len(classLabels)):
        label = classLabels[i]
        print(label)
        if not os.path.exists( os.path.join(out_data_dir, label)): os.makedirs(os.path.join(out_data_dir, label))
        
        for root, dirs, files in os.walk(os.path.join(data_dir, label)):
            cnt = 0
            for name in files:
                try:
                    filename = os.path.join(root, name)
                    img = imread(filename)
                    if CNTK:
                        filename = "ntrain.zip@/" + label + "/" + name
                    print(filename)
                    out_filename = os.path.join(out_data_dir, os.path.join(label, name))
                    out_img = getCleanedImageFromRGBImage(img)
                    out_img = cv2.resize(out_img, dsize=(rows, cols))
                    cv2.imwrite(out_filename, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

                    data.append( filename )
                    labels.append(i)

                    cnt += 1
                    if DEVELOPMENT and cnt>=2: break
                except Exception as e:
                    print(e)
                    print("error in reading file " + filename)
                    pass

    data = np.array(data)
    labels = np.array(labels)
    print(data.shape)
    print(labels.shape)
    print(data[0].shape)
    print(labels[0].shape)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)
    return X_train, X_test, y_train, y_test

def write_data_map(X, y, map_file):
    with open(map_file, 'w') as fp:
        N = len(X)
        for i in range(N):
            print("{0}\t{1}".format(X[i], y[i]), file=fp)

def write_mean_file(mean_file, rows=224, cols=224, channels=3):
    total_features = rows*cols*channels
    mean_value = "128"
    mean_file_content = '''
<?xml version="1.0" ?>
<opencv_storage>
  <Channel>''' + str(channels) + '''</Channel>
  <Row>''' + str(rows) + '''</Row>
  <Col>''' + str(rows) + '''</Col>
  <MeanImg type_id="opencv-matrix">
    <rows>1</rows>
    <cols>''' + str(total_features) + '''</cols>
    <dt>f</dt>
    <data>''' + " ".join([mean_value]*(total_features)) + '''</data>
  </MeanImg>
</opencv_storage>
'''

    with open(mean_file, 'w') as fp:
        fp.write(mean_file_content)


if __name__ == "__main__":
    data_dir = "C:\\Users\\t-anik\\Desktop\\personal\\KaggleData\\cervical-cancer\\nodupe\\"
    X_train, X_test, y_train, y_test = get_features_and_labels( os.path.join( data_dir, "train"), os.path.join( data_dir, "ntrain") )
    print("n_train  = {0}, n_test = {1}".format(len(X_train), len(X_test)))
    write_data_map(X_train, y_train, os.path.join(data_dir, "train_map.txt"))
    write_data_map(X_test, y_test, os.path.join(data_dir, "test_map.txt"))

    write_mean_file(os.path.join(data_dir, "cervix_mean.xml"))