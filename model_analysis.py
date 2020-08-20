import pvml
import numpy as np
import matplotlib.pyplot as plt
import sys

data = np.loadtxt("test.txt.gz")
X_ts = data[:, :-1]
Y_ts = data[:, -1].astype(np.int)


net = pvml.MLP.load("models/a-64-32-16-6/a-64-32-16-6-e10000-b120-lr0.001.npz")
predictions, probs = net.inference(X_ts)
accuracy = (predictions == Y_ts).mean()
print(accuracy)

m = X_ts.shape[0]
p_correct = probs[np.arange(m), Y_ts]
indices = p_correct.argsort()
print("# Worst errors")
print("Real class | Predicted class")
for i in range(10):
    print(Y_ts[indices[i]], predictions[indices[i]])

print()
print("# Confusion matrix")
confusion_matrix = np.zeros((6, 6))
for p, y in zip(predictions, Y_ts):
    confusion_matrix[y, p] += 1
confusion_matrix /= confusion_matrix.sum(1, keepdims = True)
np.savetxt(sys.stdout, 100 * confusion_matrix, fmt="%2d")

print()
freq = np.diag(confusion_matrix)
indices = freq.argsort()
print("# Hardest class")
print(indices[0], freq[indices[0]])
