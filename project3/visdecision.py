"""

INPUT:	
X: 2xn array of 2d points to visualize
Y: 1xn array of labels of X
classify: svm classifier function that takes 2xn points as input and returns 1 or -1

Visualizes a classifier's decision boundaries and 2d points with labels.
You must call plt.show() after calling this function to see the plot.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def visdecision(X, Y, classify):
    n_x = 100
    n_y = 100
    n = n_x * n_y
        
    x_min = -5
    x_max = 5
    
    y_min = -5
    y_max = 5
    
    x_lin = np.linspace(x_min, x_max, n_x)
    y_lin = np.linspace(y_max, y_min, n_y)
    x_grid, y_grid = np.array(np.meshgrid(x_lin, y_lin))
    
    x_flat = x_grid.reshape((1,n))
    y_flat = y_grid.reshape((1,n))
    grid_points = np.vstack((x_flat, y_flat))
    
    grid_classification = classify(grid_points)
    classification_to_plot = grid_classification.reshape((n_x, n_y))
    
    plt.figure()
    
    extent = [x_min, x_max, y_min, y_max]
    colors = ['red', 'blue']
    cmap = ListedColormap(colors)
    plt.imshow(classification_to_plot, extent=extent, cmap=cmap)
    
    pos = np.where(Y == 1)
    neg = np.where(Y == -1)
    
    scatter_x_pos = X[0, pos]
    scatter_y_pos = X[1, pos]
    
    scatter_x_neg = X[0, neg]
    scatter_y_neg = X[1, neg]
    
    plt.scatter(scatter_x_pos, scatter_y_pos, marker='.', color='black')
    plt.scatter(scatter_x_neg, scatter_y_neg, marker='x', color='black')