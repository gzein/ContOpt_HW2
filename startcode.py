import numpy as np
import matplotlib.pyplot as plt

def vec2mat(x, m=None, n=None):
    """
    Reshapes a vector x into an m x n matrix.
    If m and n are not given, it tries to make a square matrix.
    
    Parameters:
    x : array-like
        Input vector to be reshaped.
    m : int, optional
        Number of rows of the reshaped matrix.
    n : int, optional
        Number of columns of the reshaped matrix.
        
    Returns:
    numpy.ndarray
        Reshaped m x n matrix.
    """
    x = np.array(x)
    if m is None and n is None:
        m = int(np.sqrt(x.size))
        if m * m != x.size:
            raise ValueError("vec2mat: Cannot make a square matrix from the input vector.")
        n = m
    elif m is None or n is None:
        raise ValueError("vec2mat: Both dimensions m and n must be provided or both left as None.")
    
    return np.reshape(x, (m, n))


def plotimage(X):
    """
    Plots the image of matrix X with a colorbar.
    
    Parameters:
    X : 2D array-like
        Input matrix to be plotted.
    """
    X = np.array(X)
    m, n = X.shape
    if m != n:
        print("Warning: plotimage - Input matrix is not square.")
    
    x = np.linspace(-0.5, 0.5, m)
    y = np.linspace(-0.5, 0.5, n)
    
    plt.figure()
    plt.imshow(X, extent=[x[0], x[-1], y[0], y[-1]], aspect='equal', origin='lower')
    plt.colorbar()
    plt.axis('tight')
    plt.draw()
    plt.show()


def mat2vec(X):
    """
    Stacks the columns of a matrix X to make a vector.
    
    Parameters:
    X : 2D array-like
        Input matrix to be converted to a vector.
        
    Returns:
    numpy.ndarray
        Vector with stacked columns of X.
    """
    X = np.array(X)
    return X.flatten(order='F')  # 'F' for column-major order, similar to MATLAB's column stacking




