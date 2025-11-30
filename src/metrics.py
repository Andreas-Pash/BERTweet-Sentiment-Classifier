import numpy as np
from sklearn import metrics

import matplotlib.pyplot as plt


def confusion_matrix_heatmap(y_test, preds, labels):
    """Function to plot a confusion matrix"""
    # cm = metrics.confusion_matrix(y_test, preds, labels)
    cm = metrics.confusion_matrix(y_test, preds, labels=labels)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels( labels, rotation=45)
    ax.set_yticklabels( labels)

    for i in range(len(cm)):
        for j in range(len(cm)):
            text = ax.text(j, i, cm[i, j],
                           ha="center", va="center", color="w")

    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    b, t = plt.ylim() 
    b += 0.5  
    t -= 0.5  
    plt.ylim(b, t) # update the ylim(bottom, top) values
    plt.show()  
