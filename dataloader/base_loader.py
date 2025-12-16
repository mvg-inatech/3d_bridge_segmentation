from os.path import join
import itertools
import numpy as np
import random
from torch.utils.data import Dataset
from plyfile import PlyData
from common.filter import filter_for_range
from common.read_point_cloud import read_laspy_data, read_ply_data
from dataloader.augmentation import (
    RandomPointJitter,
    RandomPointDrop,
    RandomPointScale,
    RandomPointRotateZ,
    RandomPointFlip,
    GaussianPointNoise,
    GaussianColorNoise,
    ChromaticColorTranslation,
    RandomColorDrop,
)
from common.read_point_cloud import parse_dir_for_X_file


class SubCloud(object):
    def __init__(self, file, pos, pts, color, label):
        self.file = file
        self.pos = pos
        self.pts = pts
        self.color = color
        if label is not None:
            self.labels = label
        else:
            label = None


class SubIDX(object):
    def __init__(self, pos, idx):
        self.pos = pos
        self.idx = idx


def calculate_sub_clouds(pts, color, f, bound, label=None, min_pts=500):
    """
    Calculate sub clouds from a point cloud.

    Args:
        pts (np.array): point cloud
        color (np.array): color of the point cloud
        f (str): file name
        bound (np.array): bounding box defining max size
        label (np.array): label of the point cloud
        min_pts
    """
    sub_clouds = list()
    potentials = np.zeros((len(pts)))
    current_pos = pts[np.random.randint(0, len(potentials)), ...]
    while min(potentials) == 0:
        idx_pts = filter_for_range(pts[:, :3], bound, current_pos)
        potentials[idx_pts] += 1
        if len(idx_pts) > min_pts:
            sub_clouds.append(
                SubCloud(
                    f,
                    current_pos,
                    pts[idx_pts],
                    color[idx_pts],
                    label[idx_pts] if label is not None else None,
                )
            )
        choosen_idx = np.random.choice(np.where(potentials == np.min(potentials))[0])
        current_pos = pts[choosen_idx, :3]
    return sub_clouds


def calculate_sub_idx(pts, bound, min_pts=5000, times=1):
    """
    Calculate sub clouds from a point cloud. In this case remember only indcies
    Args:
        pts (np.array): point cloud
        bound (np.array): bounding box defining max size
        min_pts
    """
    sub_idx = list()
    potentials = np.zeros((len(pts)))
    current_pos = pts[np.random.randint(0, len(potentials)), ...]
    while min(potentials) < times:
        idx_pts = filter_for_range(pts[:, :3], bound, current_pos)
        potentials[idx_pts] += 1
        if len(idx_pts) > min_pts:
            sub_idx.append(SubIDX(current_pos, idx_pts))
        choosen_idx = np.random.choice(np.where(potentials == np.min(potentials))[0])
        current_pos = pts[choosen_idx, :3]
    return sub_idx


class SubCloudBase(Dataset):
    def __init__(self, bound, loops, root_dir, init_sub_clouds, ending):
        self.bound = bound
        self.loops = loops
        self.ending = ending
        self.files = []
        if ending == ".las":
            self.files = parse_dir_for_X_file(root_dir, ending)
        elif ending == ".ply":
            self.files = parse_dir_for_X_file(root_dir, ending)
        else:
            raise NotImplementedError(f"Ending {ending} not implemented")
        self.files = [join(root_dir, file) for file in self.files]
        if init_sub_clouds:
            self.init_sub_clouds()

    def __len__(self):
        return len(self.sub_clouds) * self.loops

    def init_sub_clouds(self):
        self.sub_clouds = list()
        for f in self.files:
            if self.ending == ".las":
                pts, color, label = read_laspy_data(f)
            elif self.ending == ".ply":
                pts, color, label = read_ply_data(f)
            else:
                raise NotImplementedError(f"Ending {self.ending} not implemented")

            pts -= np.median(pts, axis=0)

            self.sub_clouds.append(
                calculate_sub_clouds(pts, color, f, self.bound, label)
            )

            print("Done with file {}".format(f))
        self.sub_clouds = list(itertools.chain.from_iterable(self.sub_clouds))
        if len(self.sub_clouds) == 0:
            raise ValueError(
                "No point clouds loaded -> most probably wrong directory or wrong ending{}".format(
                    self.ending
                )
            )

    def transform(self, data_dict):
        """
        Transforms the data_dict by the augmentation
        """
        data_dict = RandomPointDrop()(data_dict)
        if random.random() > 0.5:
            data_dict = RandomPointJitter()(data_dict)
        if random.random() > 0.5:
            data_dict = RandomPointScale()(data_dict)
        if random.random() > 0.5:
            data_dict = RandomPointRotateZ()(data_dict)
        if random.random() > 0.5:
            data_dict = RandomPointFlip()(data_dict)
        if random.random() > 0.5:
            data_dict = GaussianPointNoise()(data_dict)
        # feats
        if random.random() > 0.5:
            data_dict = GaussianColorNoise()(data_dict)
        if random.random() > 0.5:
            data_dict = ChromaticColorTranslation()(data_dict)
        if random.random() > 0.5:
            data_dict = RandomColorDrop()(data_dict)
        return data_dict
