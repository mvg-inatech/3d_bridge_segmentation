import torch
import numpy as np
from lib.downsampling import voxel_down_label
from common.voxelize import voxelize


def sample_to_device(sample, device):
    for k, v in sample.items():
        sample[k] = v.to(device)
    return sample


def voxel_down(batch, color, label, vs=0.05):
    down = voxel_down_label(batch, color, label, vs)[0]
    batch = torch.from_numpy(down[:, :3]).float()
    color = torch.from_numpy(down[:, 4:]).float()
    label = torch.from_numpy(down[:, 3]).float()
    return batch, color, label


class VoxelDownsamplingCollator(object):
    """
    This class is used to collate the data into a batch.
    The points are downsampled CONTINIOUS!!!
    """

    def __init__(self, voxel_size):
        self.voxel_size = voxel_size

    def __call__(self, pre_batch):
        coords_list = list()
        color_list = list()
        labels_list = list()
        offsets_list = list()
        count = 0
        for sample in pre_batch:
            if sample["labels"] is None:
                sample["labels"] = np.zeros(sample["coords"].shape[0], dtype=np.int64)
            pts, color, labels = voxel_down(
                sample["coords"],
                sample["feats"],
                sample["labels"],
                vs=self.voxel_size,
            )
            coords_list.append(pts.float())
            color_list.append(color.float())
            labels_list.append(labels.long())
            count += pts.shape[0]
            offsets_list.append(count)
        batched_sample = dict()
        batched_sample["coords"] = torch.cat(coords_list)
        batched_sample["feats"] = torch.cat(color_list)
        batched_sample["labels"] = torch.cat(labels_list)
        batched_sample["offsets"] = torch.IntTensor(offsets_list)
        return batched_sample


class VoxelDiscreteCollator(object):
    """
    This class is used to collate the data into a batch.
    The points are discretized!!!
    """

    def __init__(self, voxel_size):
        self.voxel_size = voxel_size

    def __call__(self, pre_batch):
        coords_list = list()
        disc_cord_list = list()
        color_list = list()
        labels_list = list()
        offsets_list = list()
        count = 0
        for sample in pre_batch:
            if sample["labels"] is None:
                sample["labels"] = np.zeros(sample["coords"].shape[0], dtype=np.int64)
            idx, disc_coord = voxelize(sample["coords"], voxel_size=self.voxel_size)
            disc_cord_list.append(torch.from_numpy(disc_coord).float())
            coords_list.append(torch.from_numpy(sample["coords"][idx]).float())
            color_list.append(torch.from_numpy(sample["feats"][idx]).float())
            labels_list.append(torch.from_numpy(sample["labels"][idx]).long())
            count += idx.shape[0]
            offsets_list.append(count)
        batched_sample = dict()
        batched_sample["coords"] = torch.cat(coords_list)
        batched_sample["feats"] = torch.cat(color_list)
        batched_sample["labels"] = torch.cat(labels_list)
        batched_sample["grid_coord"] = torch.cat(disc_cord_list)
        batched_sample["offsets"] = torch.IntTensor(offsets_list)
        return batched_sample
