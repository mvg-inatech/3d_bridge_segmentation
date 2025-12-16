import numpy as np
import random

from common.geometry import rotate_point_cloud_yaw

######################################################################
# Point Augmentation


class RandomPointJitter():
    """
    Jitters the data by a uniform distributed random amount
        return data_dict
    """

    def __init__(self, jitter=((-1, 1), (-1, 1), (-1, 1))):
        self.jitter = jitter

    def __call__(self, data_dict):
        if "coords" in data_dict.keys():
            jitter_x = random.uniform(self.jitter[0][0], self.jitter[0][1])
            jitter_y = random.uniform(self.jitter[1][0], self.jitter[1][1])
            jitter_z = random.uniform(self.jitter[2][0], self.jitter[2][1])
            data_dict["coords"][:, 0] += jitter_x
            data_dict["coords"][:, 1] += jitter_y
            data_dict["coords"][:, 2] += jitter_z
        return data_dict


class RandomPointDrop():
    """
    Randomly drops points from the data
    """

    def __init__(self, drop_rate=0.1):
        self.drop_rate = drop_rate

    def __call__(self, data_dict):
        if "coords" in data_dict.keys():
            points_to_drop = np.random.randint(
                0,
                len(data_dict["coords"]) - 1,
                int(len(data_dict["coords"]) * self.drop_rate),
            )
            data_dict["coords"] = np.delete(data_dict["coords"], points_to_drop, axis=0)

            if "feats" in data_dict.keys():
                data_dict["feats"] = np.delete(
                    data_dict["feats"],
                    points_to_drop,
                    axis=0,
                )
            if "labels" in data_dict.keys():
                data_dict["labels"] = np.delete(
                    data_dict["labels"],
                    points_to_drop,
                    axis=0,
                )
        return data_dict


class RandomPointScale():
    """
    Randomly scale the point cloud.
    """

    def __init__(self, scale_range=(0.95, 1.05)):
        self.scale_range = scale_range

    def __call__(self, data_dict):
        if "coords" in data_dict.keys():
            scan = data_dict["coords"]
            factor = random.uniform(self.scale_range[0], self.scale_range[1])
            scan[:, :3] = scan[:, :3] * factor
        return data_dict


class RandomPointRotateZ():
    """
    Randomly rotate the point cloud arround z
    """

    def __init__(self, angle_range=(-180, 180)):
        self.angle_range = angle_range

    def __call__(self, data_dict):
        if "coords" in data_dict.keys():
            angle = random.uniform(self.angle_range[0], self.angle_range[1])
            data_dict["coords"] = rotate_point_cloud_yaw(data_dict["coords"], angle)
        return data_dict


class RandomPointFlip():
    """
    Randomly flip the point cloud (only x and y)
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, data_dict):
        if "coords" in data_dict.keys():
            if random.random() < self.prob:
                data_dict["coords"][:, 0] = -data_dict["coords"][:, 0]
            if random.random() < self.prob:
                data_dict["coords"][:, 1] = -data_dict["coords"][:, 1]
        return data_dict


class GaussianPointNoise():
    """
    Add Gaussian noise to the point cloud
    """

    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, data_dict):
        if "coords" in data_dict.keys():
            noise = np.random.normal(0, self.sigma, data_dict["coords"].shape)
            data_dict["coords"] += noise
        return data_dict


######################################################################
# Color Augmentation


class GaussianColorNoise():
    """
    Add Gaussian noise to the point cloud
    """

    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, data_dict):
        if "feats" in data_dict.keys():
            noise = np.random.normal(0, self.sigma, data_dict["feats"].shape)
            data_dict["feats"] += noise
        return data_dict


class ChromaticColorTranslation():
    """
    Add chromatic color translation to the point cloud
    """

    def __init__(self, ratio=0.01):
        self.ratio = ratio

    def __call__(self, data_dict):
        if "feats" in data_dict.keys():
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.ratio
            data_dict["feats"] = data_dict["feats"] + tr
        return data_dict


class RandomColorDrop():
    """
    Drop random colors
    """

    def __init__(self, drop_rate=0.01):
        self.drop_rate = drop_rate

    def __call__(self, data_dict):
        if "feats" in data_dict.keys():
            color_to_drop = np.random.randint(
                0,
                len(data_dict["feats"]) - 1,
                int(len(data_dict["feats"]) * self.drop_rate),
            )
            data_dict["feats"][color_to_drop] *= 0
        return data_dict
