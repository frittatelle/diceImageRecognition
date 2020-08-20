import os
import matplotlib.pyplot as plt
import numpy as np
import image_features
import cv2
import pvml




def extract_neural_features(img, net):
    activations = net.forward(img[None, :, :, :])
    features = activations[-3]
    features = features.reshape(-1)
    return features

def process_dir(dir, net):
    filenames = os.listdir(dir)
    img_features = []
    img_labels = []
    for filename in filenames:
        img_name = dir + "/" + filename
        img =  cv2.imread(img_name)
        img = cv2.copyMakeBorder( img, 48, 48, 48, 48, cv2.BORDER_CONSTANT)
        img = np.array(img)
        img_feature = extract_neural_features(img, net)
        print(img_feature.shape)
        img_label = filename[0]
        img_features.append(img_feature)
        img_labels.append(img_label)
    X = np.stack(img_features, 0)
    Y = np.array(img_labels).astype(np.int)
    Y = Y - 1
    return X, Y

cnn = pvml.PVMLNet.load("pvmlnet.npz")

X, Y = process_dir("train", cnn)
print(X.shape, Y.shape)
data = np.concatenate([X, Y[:,None]], 1)
np.savetxt("train_neural.txt.gz", data)

# X, Y = process_dir("test", cnn)
# print(X.shape, Y.shape)
# data = np.concatenate([X, Y[:,None]], 1)
# np.savetxt("test_neural.txt.gz", data)
#
# X, Y = process_dir("validation", cnn)
# print(X.shape, Y.shape)
# data = np.concatenate([X, Y[:,None]], 1)
# np.savetxt("validation_neural.txt.gz", data)
