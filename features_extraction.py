import os
import matplotlib.pyplot as plt
import numpy as np
import image_features
import cv2

def process_dir(dir):
    filenames = os.listdir(dir)
    img_features = []
    img_labels = []
    for filename in filenames:
        img_name = dir + "/" + filename
        img = plt.imread(img_name)
        # img_feature = image_features.cooccurrence_matrix(img)
        # img_feature = image_features.color_histogram(img)
        # img_feature = img_feature.reshape(-1)
        # img_features.append(img_feature)
        feature_0 = image_features.cooccurrence_matrix(img)
        feature_0 = feature_0.reshape(-1)
        feature_1 = image_features.color_histogram(img)
        feature_1 = feature_1.reshape(-1)
        features = np.concatenate((feature_0, feature_1),axis = 0)
        img_features.append(features)
        img_label = filename[0]
        img_labels.append(img_label)
    X = np.stack(img_features, 0)
    Y = np.array(img_labels).astype(np.int)
    Y = Y - 1
    return X, Y



X, Y = process_dir("train")
print(X.shape, Y.shape)
data = np.concatenate([X, Y[:,None]], 1)
np.savetxt("train_coocmx_ch.txt.gz", data)

X, Y = process_dir("test")
print(X.shape, Y.shape)
data = np.concatenate([X, Y[:,None]], 1)
np.savetxt("test_coocmx_ch.txt.gz", data)

X, Y = process_dir("validation")
print(X.shape, Y.shape)
data = np.concatenate([X, Y[:,None]], 1)
np.savetxt("validation_coocmx_ch.txt.gz", data)
