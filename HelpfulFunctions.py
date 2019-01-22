# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 17:43:45 2017

@author: Dominic
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def print_cm(cm,labels,plt):      

    print(cm)
    plt.figure(figsize=(10, 10))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap=plt.cm.Greys)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

#from sklearn.metrics import confusion_matrix
#from sklearn.linear_model import LogisticRegression

def featureSpacePlot(Xname,Yname,data,y,classifier,plt):

    h = .01  # step size in the mesh

    X = data[Xname]
    Y = data[Yname]
        
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
                            
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X.min() - .05, X.max() + .05
    y_min, y_max = Y.min() - .05, Y.max() + .05
    
    ######################################
    
    nx = (x_max - x_min)/ h
    ny = (y_max - y_min)/ h
    
    g1 = np.linspace(x_min, x_max, nx)
    g2 = np.linspace(y_min, y_max, ny)

    xx, yy = np.meshgrid(g1,g2)

    ######################################
    
#    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    colors = np.resize(y,[len(y),1])

    # Plot also the training points
    plt.scatter(X,Y, c=y, edgecolors='k', cmap=cmap_bold, alpha = 1.0)

    plt.pcolormesh(xx, yy, Z, cmap=cmap_bold, alpha=0.1)

    plt.xlabel(Xname)
    plt.ylabel(Yname)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    