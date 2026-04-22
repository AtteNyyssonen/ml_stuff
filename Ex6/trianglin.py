import numpy as np


def trianglin(P1, P2, x1, x2):
    """
    :param P1: Projection matrix for image 1 with shape (3,4)
    :param P2: Projection matrix for image 2 with shape (3,4)
    :param x1: Image coordinates for a point in image 1
    :param x2: Image coordinates for a point in image 2
    :return X: Triangulated world coordinates
    """
    
    # Form A and get the least squares solution from the eigenvector 
    # corresponding to the smallest eigenvalue
    ##-your-code-starts-here-##
    x1P1 = np.cross(x1, P1.T).T
    x2P2 = np.cross(x2, P2.T).T
    A = np.vstack((x1P1, x2P2))

    # SVD as in the last task to solve the homogeneous linear system
    U, S, V = np.linalg.svd(A, full_matrices=True)
    X = V[-1, :]
    ##-your-code-ends-here-##
    
    return X
