from tools import *


def save_train_graph(path, data) -> None:
    os.mkdir(os.path.join(path, "graph"))
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    xaxis = np.arange(1,data["epoch"]+1)
    for log in data.keys():
        if log != "epoch":
            ax.plot(xaxis,data[log],label=log)
    ax.set_title("Training Log", fontsize=18)
    ax.set_xlabel("Epochs", fontsize=12)
    ax.legend()
    plt.savefig(os.path.join(path, "training_graph_log"))
    