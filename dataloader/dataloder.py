import os
from torch.utils.data import DataLoader
from dataloader.loader_utils import VoxelDiscreteCollator, VoxelDownsamplingCollator
from dataloader.semantic_bridge_loader import SemanticBridgeLoader


def get_dataloader(config, train=True):
    # train or val
    if train:
        apply_transform = True
        split = "train"
        loops = config["loops"]
    else:
        apply_transform = False
        split = "val"
        loops = 1
    # collator
    if config["collator"] == "point":
        voxel_collator = VoxelDownsamplingCollator(config["voxel_size"])
    elif config["collator"] == "voxel":
        voxel_collator = VoxelDiscreteCollator(config["voxel_size"])
    else:
        raise NotImplementedError(
            "Collator {} not implemented".format(config["collator"])
        )
    # dataset
    if config["dataset"] == "semanticbridge":
        loader = SemanticBridgeLoader(
            os.path.join(config["data_dir"], split),
            config["bound"],
            apply_transform=apply_transform,
            loops=loops,
            ending=config["ending"],
        )
    else:
        raise NotImplementedError(
            "Dataset {} not implemented - only supported: semanticbridge".format(
                config["dataset"]
            )
        )
    dataloader = DataLoader(
        loader,
        batch_size=config["train"]["bs"],
        shuffle=True,
        num_workers=0,
        collate_fn=voxel_collator,
    )
    return dataloader
