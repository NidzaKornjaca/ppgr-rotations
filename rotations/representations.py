import math
import numpy as np

def rot_x_global(angle):
    """
    Return matrix for rotation
    around the global x axis for an angle
    expressed in radians.
    """
    return np.array([
        [1, 0, 0],
        [0, math.cos(angle), -math.sin(angle)],
        [0, math.sin(angle), math.cos(angle)]
    ])


def rot_y_global(angle):
    """
    Return matrix for rotation around the y axis for
    an angle expressed in radians.
    """
    return np.array([
        [math.cos(angle), 0, math.sin(angle)],
        [0, 1, 0],
        [-math.sin(angle), 0, math.cos(angle)]
    ])


def rot_z_global(angle):
    """
    Return matrix for rotation around the z axis for
    an angle expressed in radians.
    """
    return np.array([
        [math.cos(angle), -math.sin(angle), 0],
        [math.sin(angle), math.cos(angle), 0],
        [0, 0, 1]
    ])


# 1-a
def euler2a(phi, theta, psy):
    """
    Returns the matrix transform for the rotation
    around the axi x, y, z by the angles phy, theta
    and psy respectively.
    """
    rot_x = rot_x_global(phi)
    rot_y = rot_y_global(theta)
    rot_z = rot_z_global(psy)
    return rot_z @ rot_y @ rot_x


def get_eig(matrix):
    """
    Get eigenvector for lambda = 1
    matrix @ vector = vector
    matrix @ vec - vec = 0
    (matrix - E) @ vec = 0
    """
    ww, v = np.linalg.eig(matrix)
    return v[:, np.isclose(ww, 1)]


# 1-b
def axis_angle(rot_matrix):
    """
    For a given rotation matrix,
    calculates the equivalent axis angle rotation.

    :returns: tuple with the axis vector and rotation angle
    """
    p = get_eig(rot_matrix).flatten()
    # get vector normal to p
    u = np.array([
        -p[1],
        p[0],
        0
    ])
    u_p = rot_matrix @ u
    angle = math.acos(
        np.real(np.dot(u.T, u_p)) / (
            np.linalg.norm(u) * np.linalg.norm(u_p)
        )
    )
    if np.dot(u.T, np.cross(u_p.T, p.T).T) < 0:
        p = -p
    return (p, angle)


# 1 - c
def rodrigez(axis_i, angle):
    axis = np.array([axis_i])
    axis_norm = np.linalg.norm(axis)
    normed = axis / axis_norm
    axis_mat = normed.T @ normed
    p_x = np.array([
        [0, -normed[0][2], normed[0][1]],
        [normed[0][2], 0, -normed[0][0]],
        [-normed[0][1], normed[0][0], 0]
    ])
    return (
        axis_mat +
        math.cos(angle) * (np.eye(3, 3) - axis_mat) +
        math.sin(angle) * p_x
    )


# 1 - d
def a2euler(rotation_matrix):
    """
    For the matrix rotation_matrix calculates the euler angles.
    """
    if rotation_matrix[2][0] < 1:
        if rotation_matrix[2][0] > -1:  # unique solution
            psy = math.atan2(
                rotation_matrix[1][0], rotation_matrix[0][0]
            )
            theta = math.asin(-rotation_matrix[2][0])
            phi = math.atan2(
                rotation_matrix[2][1], rotation_matrix[2][2]
            )
        else:  # solution not unique - Ox3 = -Oz
            psy = math.atan2(
                -rotation_matrix[0][1], rotation_matrix[1][1]
            )
            theta = math.pi / 2
            phi = 0
    else:
        psy = math.atan2(
            -rotation_matrix[0][1], rotation_matrix[1][1]
        )
        theta = -math.pi / 2
        phi = 0
    return (phi, theta, psy)  # rot x, rot y, rot z

