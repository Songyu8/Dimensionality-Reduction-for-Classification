import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import  confusion_matrix


def plot_data(inputs, labels, sample_icon, sample_color, title):
    """ Plot dataset """
    x0, x1 = zip(*inputs)
    uniqe_lbls, dict_data = set(labels), {}
    plt.figure()

    for label in uniqe_lbls:
        dict_data[label] = []

    for i in range(len(labels)):
        dict_data[labels[i]].append(inputs[i])

    for label, color in zip(dict_data.keys(), sample_color):
        x0, x1 = zip(*dict_data[label])
        plt.plot(x0, x1, sample_icon, color=color, label=f'class {int(label)}')

    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.title(title)
    plt.legend()



def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    :param y_true: True labels
    :param y_pred: Predicted labels
    :param classes: List of class names
    :param normalize: If True, normalize the confusion matrix
    :param title: Title for the plot
    :param cmap: Colormap for the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
