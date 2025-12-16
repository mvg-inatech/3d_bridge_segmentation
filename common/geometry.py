import numpy as np

############################################################################
# TRANSFORMATIONS


def get_Rz_matrix(rad):
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    R = np.array(((cos_a, -sin_a, 0), (sin_a, cos_a, 0), (0, 0, 1)))
    return R


def rotate_point_cloud_yaw(pc, theta):
    rad = np.radians(theta)
    rot = get_Rz_matrix(rad)
    new_pc = np.copy(pc)
    new_pc[:, :3] = pc[:, :3] @ rot
    return new_pc
