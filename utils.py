import numpy as np
import matplotlib.pyplot as plt


def plot_clustering(data, y, labels, title=None):
    x_min, x_max = np.min(data, axis=0), np.max(data, axis=0)
    data = (data - x_min) / (x_max - x_min)

    plt.figure(figsize=(5, 5))
    for i in range(data.shape[0]):
        plt.text(
            data[i, 0],
            data[i, 1],
            str(int(y[i])),
            color=plt.cm.rainbow(labels[i] / 15.0),
            fontdict={"weight": "bold", "size": 9},
        )

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('t-SNE-{}.pdf'.format(title), dpi=600)
    return