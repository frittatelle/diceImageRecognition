from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("train_coocmx_ch.txt.gz")
X_tr = data[:, :-1]
Y_tr = data[:, -1]

data = np.loadtxt("test_coocmx_ch.txt.gz")
X_ts = data[:, :-1]
Y_ts = data[:, -1]

tr_accs = []
ts_accs = []
for deg in range(20):
    clf = OneVsRestClassifier(SVC(kernel='poly', degree=deg)).fit(X_tr, Y_tr)
    Y_tr_pred = clf.predict(X_tr)
    Y_ts_pred = clf.predict(X_ts)
    tr_acc = (Y_tr_pred == Y_tr).mean()
    ts_acc = (Y_ts_pred == Y_ts).mean()
    print(deg, tr_acc, ts_acc)
    tr_accs.append(tr_acc * 100)
    ts_accs.append(ts_acc * 100)

tr_accs = np.array(tr_accs)
ts_accs = np.array(ts_accs)
report_name = "models/KSVM/report_coocmx_ch.txt"
data = np.zeros(20, dtype=[('deg', 'int'), ('tr_accs', 'float'), ('ts_accs', 'float')])
data['deg'] = range(20)
data['tr_accs'] = tr_accs
data['ts_accs'] = ts_accs
np.savetxt(report_name, data, fmt="%d %.2f %.2f")
