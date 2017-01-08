import numpy as np
from scipy.ndimage import imread
import os

def get_features_and_labels(data_dir):
    classLabels = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    labels = []
    data = []
    for i in range(len(classLabels)):
        label = classLabels[i]
        print(label)
        for root, dirs, files in os.walk(os.path.join(data_dir, label)):
            for name in files:
                #print((os.path.join(root, name)))
                data.append( imread(os.path.join(root, name)) )
                labels.append(i)

if __name__ == '__main__':
    Data_Dir = 'C:\\Users\\t-anik\\Desktop\\personal\\KaggleData'

    # Process data into feature and label arrays
    get_features_and_labels(os.path.join(Data_Dir, 'train'))

