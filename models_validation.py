import pvml
import numpy as np
import matplotlib.pyplot as plt
import os

def name_net(net, epochs, batch_size, lr):
    net_name = "a"
    for layer in net:
        net_name = net_name + "-" + str(layer)
        dir_name = net_name
    net_name = net_name + "-e" + str(epochs)
    net_name = net_name + "-b" + str(batch_size)
    net_name = net_name + "-lr" + str(lr)
    net_name = net_name + ".npz"
    return net_name, dir_name

# Data loading
data = np.loadtxt("train.txt.gz")
X_tr = data[:, :-1]
Y_tr = data[:, -1].astype(np.int)

data = np.loadtxt("validation.txt.gz")
X_val = data[:, :-1]
Y_val = data[:, -1].astype(np.int)


########## Validation table ###########

# cooccurrence_matrix, 1st STEP
# nets_list = [[X_tr.shape[1], 6],
#         [X_tr.shape[1], 32, 6],
#         [X_tr.shape[1], 32, 16, 6]]
# epochs_list = [10, 100, 500, 1000]
# batch_size_list = [80, 120]
# lr_list = [1e-2, 1e-3, 1e-4]

# cooccurrence_matrix, (64, 32, 6), 2nd STEP
# nets_list = [[X_tr.shape[1], 32, 6]]
# epochs_list = [1000, 5000, 10000]
# batch_size_list = [80, 120]
# lr_list = [1e-2, 1e-3]

# cooccurrence_matrix, (64, 32, 16, 6), 2nd STEP
nets_list = [[X_tr.shape[1], 32, 16, 6]]
epochs_list = [1000, 5000, 10000]
batch_size_list = [80, 120]
lr_list = [1e-2, 1e-3]

# color_histogram, (64, 32, 16, 6)
# nets_list = [[X_tr.shape[1], 32, 16, 6]]
# epochs_list = [1000]
# batch_size_list = [80, 120]
# lr_list = [1e-2, 1e-3]


# Training + report
for net in nets_list:
    model_tr_accs = []
    model_val_accs = []
    net_names = []
    for epochs in epochs_list:
        for batch_size in batch_size_list:
            for lr in lr_list:
                mlp = pvml.MLP(net)
                net_name, dir_name = name_net(net, epochs, batch_size, lr)
                print()
                print(net_name)
                tr_accs = []
                val_accs = []
                plt.ion()
                for epoch in range(epochs):
                    steps = X_tr.shape[0] // batch_size
                    mlp.train(X_tr, Y_tr, lr=lr, batch=batch_size, steps=steps)
                    predictions, probs = mlp.inference(X_tr)
                    tr_acc = (predictions == Y_tr).mean()
                    tr_accs.append(tr_acc * 100)
                    predictions, probs = mlp.inference(X_val)
                    val_acc = (predictions == Y_val).mean()
                    val_accs.append(val_acc * 100)
                    print(epoch, tr_acc, val_acc)
                    # plt.clf()
                    # plt.plot(tr_accs)
                    # plt.plot(val_accs)
                    # plt.legend(["train", "validation"])
                    # plt.pause(0.01)
                try:
                    os.makedirs("models/" + dir_name)
                except:
                    pass
                mlp.save("models/" + dir_name +  "/" + net_name)
                model_tr_accs.append(tr_acc * 100)
                model_val_accs.append(val_acc * 100)
                net_names.append(net_name)

    net_names = np.array(net_names)
    model_tr_accs = np.array(model_tr_accs)
    model_val_accs = np.array(model_val_accs)
    report_name = "models/" + dir_name + "/report.txt"
    data = np.zeros(net_names.size, dtype=[('net_names', 'U32'), ('model_tr_accs', 'float'), ('model_val_accs', 'float')])
    data['net_names'] = net_names
    data['model_tr_accs'] = model_tr_accs
    data['model_val_accs'] = model_val_accs
    np.savetxt(report_name, data, fmt="%s %.2f %.2f")
