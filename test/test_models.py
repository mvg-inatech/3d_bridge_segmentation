import sys
from os.path import dirname, abspath, join

sys.path.insert(0, abspath(join(dirname(__file__), '..')))

import torch
import numpy as np
from common.voxelize import voxelize
from models.spconv_unet import SpConvUNet
from models.point_transformer_v2 import PointTransformerV2
from models.kp_conv import KPFCNN
from dataloader.loader_utils import sample_to_device
from common.parser import yaml_cfg_to_class


def get_random_batch():
    sim_coords = np.random.randn(6000, 3)
    sim_feats = np.random.randn(6000, 3)
    voxel_size = 0.05
    idx, disc_coord = voxelize(sim_coords, voxel_size=voxel_size)

    batched_sample = dict()
    batched_sample["coords"] = torch.from_numpy(sim_coords[idx]).float()
    batched_sample["feats"] = torch.from_numpy(sim_feats[idx]).float()
    batched_sample["grid_coord"] = torch.from_numpy(disc_coord).float()
    batched_sample["offsets"] = torch.IntTensor(list([len(idx)]))

    return batched_sample


def test_sp_conv_unet():
    batched_sample = get_random_batch()
    device = torch.device("cuda")
    sp_config = yaml_cfg_to_class("./config/sp_conv_bridge.yml")
    model = SpConvUNet(sp_config)
    model = model.eval().to(device)
    out = model(sample_to_device(batched_sample, device))
    print("SPConv ... Done")


def test_pt_v2():
    batched_sample = get_random_batch()
    device = torch.device("cuda")
    pt_config = yaml_cfg_to_class("./config/pt_v2_bridge.yml")
    model = PointTransformerV2(pt_config)
    model = model.eval().to(device)
    out = model(sample_to_device(batched_sample, device))
    print("PTv2 ... Done")


def test_kp_conv():
    batched_sample = get_random_batch()
    device = torch.device("cuda")
    kp_config = yaml_cfg_to_class("./config/kp_conv_bridge.yml")
    model = KPFCNN(kp_config)
    model = model.eval().to(device)
    out = model(sample_to_device(batched_sample, device))
    print("KPConv ... Done")


if __name__ == "__main__":
    test_sp_conv_unet()
    test_pt_v2()
    test_kp_conv()
