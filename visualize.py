import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.cm import get_cmap

# データ読み込み
no_res, use_res = [], []
for i in range(4):
    with open(f"history/no_res_{i*3+3:02d}.dat", "rb") as fp:
        no_res.append(pickle.load(fp))
    with open(f"history/use_res_{i*3+3:02d}.dat", "rb") as fp:
        use_res.append(pickle.load(fp))

xlabels = np.arange(100) + 1
cmap = get_cmap("Set1")


plt.subplot(2, 1, 1)
for i in range(len(no_res)):
    plt.plot(xlabels, no_res[i]["val_acc"], color=cmap(i), label="No_res # blocks="+str(i*3+3))
plt.legend()
plt.ylim((0.6, 0.9))

plt.subplot(2, 1, 2)
for i in range(len(no_res)):
    plt.plot(xlabels, use_res[i]["val_acc"],  color=cmap(i), label="Use_res # blocks="+str(i*3+3))
plt.legend()
plt.ylim((0.6, 0.9))

plt.suptitle("Validation accuracy")
plt.show()