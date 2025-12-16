import numpy as np


def filter_for_range(
    pts: np.ndarray, bound: np.ndarray, pos: np.ndarray = None
) -> np.ndarray:
    """
    Returns indicies of all points within given range relative to position,
    if given or mean otherwise
        :param pts: pointcloud points as a `np.array`
        :param bound: boundings in all directions as a `np.array` [3]
        :param pos: position of scanner for instance
        :returns:
        - idx: idx of points in defined range as a `np.array` [n]
    """
    # 3D case
    assert pts.shape[1] >= 3, "Points should be at least 3D with x,y,z,..."
    
    xyz = pts[:, :3]
    if pos is not None:
        mean = pos.reshape((3))
    else:
        mean = np.median(xyz, axis=0)

    threshold_x = bound[0]
    threshold_y = bound[1]
    threshold_z = bound[2]

    x_filt = np.logical_and(
        (xyz[:, 0] < mean[0] + threshold_x),
        (xyz[:, 0] > mean[0] - threshold_x),
    )
    y_filt = np.logical_and(
        (xyz[:, 1] < mean[1] + threshold_y),
        (xyz[:, 1] > mean[1] - threshold_y),
    )
    z_filt = np.logical_and(
        (xyz[:, 2] < mean[2] + threshold_z),
        (xyz[:, 2] > mean[2] - threshold_z),
    )

    filter = np.logical_and(x_filt, y_filt)
    filter = np.logical_and(filter, z_filt)
    idx = np.argwhere(filter).flatten()

    return idx
