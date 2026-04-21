import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D


def nodes(x, y, z):
    """
    Create a 3D grid of nodes.

    Parameters:
    x (int): Number of nodes in the x-direction.
    y (int): Number of nodes in the y-direction.
    z (int): Number of nodes in the z-direction.

    Returns:
    np.ndarray: A 3D array of shape (x, y, z) containing the coordinates of the nodes.
    """
    return np.array([x, y, z])

def plot_nodes(nodes):
    """
    Plot the nodes in a 3D space.

    Inputs:
    nodes (np.ndarray): A 3D array containing the coordinates of the nodes.

    returns:
    A 3D plot of the nodes with node numbers.
    """
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(nodes)):
        x, y, z = nodes[i]
        ax.scatter(x, y, z, s=20)
        ax.text(x, y, z, f'  {i+1}', fontsize=8)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        # ax.set_zlim(0, 20)
    plt.show()


def plot_nodes2d(nodes):
    """
    Plot the nodes in 2D space (x-y plane).


    Inputs:
    nodes (np.ndarray): A 3D array containing the coordinates of the nodes.

    returns:
    A 2D plot of the nodes with node numbers.
    """
    fig, ax = plt.subplots(figsize=(20, 20))

    for i in range(len(nodes)):
        x, y, z = nodes[i]
        ax.scatter(x, y, s=50)
        ax.text(x, y, f'  {i+1}', fontsize=8)
    
    ax.set_xlabel('Length [m]')
    ax.set_ylabel('Length [m]')
    ax.set_aspect('equal')    
    ax.grid(True)
    plt.show()


    
def degrees_of_freedom(nodes):
    """
    Assign degrees of freedom to each node.

    Inputs:
    nodes (np.ndarray): A 3D array containing the coordinates of the nodes.

    Returns:
    dict: A dictionary mapping each node index to its degrees of freedom.
    """
    dof = {}
    for i in range(len(nodes)):
        dof[i] = [f'ux{i+1}', f'uy{i+1}', f'uz{i+1}', f'rot_x{i+1}', f'rot_y{i+1}', f'rot_z{i+1}'] 
        dof[f'dof_{i+1}'] = list(range(i*6, (i+1)*6))
    return dof