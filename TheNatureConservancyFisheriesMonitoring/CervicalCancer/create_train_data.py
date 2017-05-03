DATA_PATH = 'C:\\Users\\t-anik\\Desktop\\personal\\KaggleData\\cervical-cancer\\'
TRAIN_TEST_DATA_FILE = DATA_PATH + "train_test_data_file.npz"

import numpy as np
import cv2, os
npzfile = np.load(TRAIN_TEST_DATA_FILE)
X_train, X_val, y_train, y_val = npzfile['X_train'], npzfile['X_val'], npzfile['y_train'], npzfile['y_val']
X_test, filenames =  npzfile['X_test'],  npzfile['filenames']

n_train, n_val, n_test = len(X_train), len(X_val), len(X_test)
print("n_train={0}, n_val={1} and n_test={2}".format(n_train, n_val, n_test))
   
def write_data_map(X, y, map_file, out_data_dir):
    if not os.path.exists(out_data_dir): os.makedirs(out_data_dir)
    with open(map_file, 'w') as fp:
        N = len(X)
        for i in range(N):
            out_img = X[i].reshape((224, 224, 3))
            out_filename = os.path.join(out_data_dir, str(i) + ".png")
            cv2.imwrite(out_filename, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
            print("{0}\t{1}".format(out_filename, y[i]), file=fp)

data_dir = "C:\\Users\\t-anik\\Desktop\\personal\\KaggleData\\cervical-cancer\\augm_data\\"
write_data_map(X_train, y_train, os.path.join(data_dir, "train_map.txt"),os.path.join(data_dir, "train") )
write_data_map(X_val, y_val, os.path.join(data_dir, "test_map.txt"),os.path.join(data_dir, "test") )