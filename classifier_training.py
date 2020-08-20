import pvml
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("train_coocmx_ch.txt.gz")
X_tr = data[:, :-1]
Y_tr = data[:, -1].astype(np.int)

data = np.loadtxt("test_coocmx_ch.txt.gz")
X_ts = data[:, :-1]
Y_ts = data[:, -1].astype(np.int)

mlp = pvml.MLP([X_tr.shape[1], 6])
# mlp = pvml.MLP([X_tr.shape[1], 32, 6])
# mlp = pvml.MLP([X_tr.shape[1], 32, 16, 6])


epochs = 1000
batch_size = 120
lr = 0.01

tr_accs = []
ts_accs = []
plt.ion()
for epoch in range(epochs):
    steps = X_tr.shape[0] // batch_size
    mlp.train(X_tr, Y_tr, lr=lr, batch=batch_size, steps=steps)
    predictions, probs = mlp.inference(X_tr)
    tr_acc = (predictions == Y_tr).mean()
    tr_accs.append(tr_acc * 100)
    predictions, probs = mlp.inference(X_ts)
    ts_acc = (predictions == Y_ts).mean()
    ts_accs.append(ts_acc * 100)
    print(epoch, tr_acc, ts_acc)
    # plt.clf()
    # plt.plot(tr_accs)
    # plt.plot(ts_accs)
    # plt.legend(["train", "test"])
    # plt.pause(0.01)

file = "net.npz"
mlp.save(file)

plt.ioff()
plt.show()
