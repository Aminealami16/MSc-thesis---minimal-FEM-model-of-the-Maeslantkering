import numpy as np
import sys
import os
sys.path.append(os.path.abspath('..'))
from scripts import timoshenko_model as tm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D





def plot_elements(elements):

    """
    Plot elements in 3D.

    Inputs:
    elements: List of tuples, where each tuple contains the coordinates of the two nodes and the element matrices (M_e, K_e).
              Each tuple should be in the form (n1, n2, M_e, K_e), where n1 and n2 are 3D coordinates of the nodes.

    
    returns:
    A 3D plot of the elements with node numbers at the midpoints.   
    """
    fig = plt.figure(figsize=(25, 25))
    ax = fig.add_subplot(111, projection='3d')

    for i, element in enumerate(elements):
        n1, n2, _, _ = element
        x = [n1[0], n2[0]]
        y = [n1[1], n2[1]]
        z = [n1[2], n2[2]]
        ax.plot(x, y, z, 'b-')
        
        # Plot nodes
        ax.scatter(*n1, color='red', s=50)
        ax.scatter(*n2, color='red', s=50)
        
        # Add element number at midpoint
        mid_x = (n1[0] + n2[0]) / 2
        mid_y = (n1[1] + n2[1]) / 2
        mid_z = (n1[2] + n2[2]) / 2
        ax.text(mid_x, mid_y, mid_z, str(i), fontsize=15, color='black')
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_zlim(0, 100)
    plt.show()
    
def plot_elements2d(elements):
    """
    Plot elements in 2D. 

    Inputs:
    elements: List of tuples, where each tuple contains the coordinates of the two nodes and the element matrices (M_e, K_e).
              Each tuple should be in the form (n1, n2, M_e, K_e), where n1 and n2 are 2D coordinates of the nodes. 

    returns:
    A 2D plot of the elements with node numbers at the midpoints.
    """
    fig, ax = plt.subplots(figsize=(20, 20))
    
    for i, element in enumerate(elements):
        n1, n2, _, _ = element
        x = [n1[0], n2[0]]
        y = [n1[1], n2[1]]
        ax.plot(x, y, 'b-')
        
        # Plot nodes
        ax.scatter(*n1[:2], color='red', s=50)
        ax.scatter(*n2[:2], color='red', s=50)
        
        # Add element number at midpoint
        mid_x = (n1[0] + n2[0]) / 2
        mid_y = (n1[1] + n2[1]) / 2
        ax.text(mid_x, mid_y, str(i), fontsize=15, color='black')
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_aspect('equal')
    plt.show()


def elements(n1, n2, ep_K, ep_M):
    """
    Create an element connecting two nodes.

    Parameters:
    n1 (np.ndarray): Coordinates of the first node.
    n2 (np.ndarray): Coordinates of the second node.

    Returns:
    tuple: A tuple containing the coordinates of the two nodes.
    """
    ex = [n1[0], n2[0]]
    ey = [n1[1], n2[1]]
    ez = [n1[2], n2[2]]
    eo = [0, 0 , 1]

    M_e, K_e = tm.T_element(ex, ey, ez, eo, ep_K, ep_M)

    return n1, n2, M_e, K_e


def elements_added_mass(n1, n2, ep_K, ep_M):
    """
    Create an element connecting two nodes.

    Parameters:
    n1 (np.ndarray): Coordinates of the first node.
    n2 (np.ndarray): Coordinates of the second node.

    Returns:
    tuple: A tuple containing the coordinates of the two nodes.
    """
    ex = [n1[0], n2[0]]
    ey = [n1[1], n2[1]]
    ez = [n1[2], n2[2]]
    eo = [0, 0 , 1]


    M_e, K_e = tm.T_element_added_mass_retaining_wall(ex, ey, ez, eo, ep_K, ep_M)

    return n1, n2, M_e, K_e



def expand_eigenvectors(eigvecs_reduced, keep_dofs, total_dofs):
    """
    Expand reduced eigenvectors to full DOF size by inserting zeros
    at constrained DOFs.
    
    Inputs::
    eigvecs_reduced : array, shape (num_free_dofs, num_modes)
        Eigenvectors from reduced system
    keep_dofs : list or array
        Indices of free DOFs kept in reduced system
    total_dofs : int
        Total number of DOFs in the full system
    
    Returns:
    --------
    eigvecs_full : array, shape (total_dofs, num_modes)
        Eigenvectors in full DOF space, zeros at constrained DOFs
    """
    num_modes = eigvecs_reduced.shape[1]
    eigvecs_full = np.zeros((total_dofs, num_modes))
    
    for j, dof in enumerate(keep_dofs):
        eigvecs_full[dof, :] = eigvecs_reduced[j, :]
    
    return eigvecs_full



def extract_displacement(arr, keep=3, skip=3):
    """
    Extract displacement DOFs from a 2D array of eigenvectors.

    Inputs:
    arr: 2D numpy array where each column is an eigenvector
    keep: number of rows to keep in each cycle
    skip: number of rows to skip in each cycle

    Returns a compact array of displacement DOFs
    """
    n_rows = arr.shape[0]
    step = keep + skip

    # Generate the indices once
    idx = np.hstack([
        np.arange(i, min(i + keep, n_rows))
        for i in range(0, n_rows, step)
    ])

    # Apply same row indices to ALL columns → returns a compact matrix
    return arr[idx, :]


def extract_rotation(arr, keep=3, skip=3):
    """
    arr: 2D numpy array where each column is an eigenvector
    keep: number of rotational DOFs to keep in each cycle (default 3: θx, θy, θz)
    skip: number of translational DOFs to skip in each cycle (default 3: ux, uy, uz)
    
    Returns a compact array of rotational DOFs
    """
    n_rows = arr.shape[0]
    step = keep + skip

    # Generate indices for rotations
    idx = np.hstack([
        np.arange(i + skip, min(i + step, n_rows))  # start after skip rows
        for i in range(0, n_rows, step)
    ])

    return arr[idx, :]

def set_equal_aspect(fig, X, Y, Z):
    max_range = max(
        max(X) - min(X),
        max(Y) - min(Y),
        max(Z) - min(Z)
    ) / 2.0

    mid_x = (max(X) + min(X)) / 2.0
    mid_y = (max(Y) + min(Y)) / 2.0
    mid_z = (max(Z) + min(Z)) / 2.0

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[mid_x - max_range, mid_x + max_range]),
            yaxis=dict(range=[mid_y - max_range, mid_y + max_range]),
            zaxis=dict(range=[mid_z - max_range, mid_z + max_range]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1)
        )
    )