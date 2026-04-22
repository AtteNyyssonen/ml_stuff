import numpy as np


def camcalibDLT(x_world, x_im):
    """
    :param x_world: World coordinatesm with shape (point_id, coordinates)
    :param x_im: Image coordinates with shape (point_id, coordinates)
    :return P: Camera projection matrix with shape (3,4)
    """

    # Create the matrix A 
    ##-your-code-starts-here-##
    A = []
    num_of_points = x_world.shape[0]
    for i in range(num_of_points):
        x, y, z, h = x_world[i]
        x_t, y_t, h_t = x_im[i]
        A_i = np.array([
            [0, 0, 0, 0, x, y, z, h, -y_t*x, -y_t*y, -y_t*z, -y_t*h],
            [x, y, z, h, 0, 0, 0, 0, -x_t*x, -x_t*y, -x_t*z, -x_t*h]
        ])
        A.append(A_i)
    A = np.concatenate(A, axis=0)
    ##-your-code-ends-here-##
    
    # Perform homogeneous least squares fitting.
    # The best solution is given by the eigenvector of
    # A.T*A with the smallest eigenvalue.
    ##-your-code-starts-here-##
    # The smallest eigenvalue A.T*A is equivalent to the column of V with the smallest singluar value in SVD
    U, S, V = np.linalg.svd(A, full_matrices=True)
    ev = V[-1, :]
    ##-your-code-ends-here-##
    
    # Reshape the eigenvector into a projection matrix P
    P = np.reshape(ev, (3, 4))  # here ev is the eigenvector from above
    #P = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=float)  # remove this and uncomment the line above
    
    return P
