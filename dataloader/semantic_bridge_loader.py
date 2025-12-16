from copy import deepcopy
import numpy as np

from dataloader.base_loader import SubCloudBase


class SemanticBridgeLoader(SubCloudBase):
    """ """

    def __init__(
        self,
        root_dir,
        bound,
        apply_transform=False,
        init_sub_clouds=True,
        loops=1,
        ending=".las",
    ):
        super().__init__(bound, loops, root_dir, init_sub_clouds, ending)
        self.root_dir = root_dir
        self.apply_transform = apply_transform

        self.labels = list(
            [
                "unlabaled",
                "underground",
                "high_vegetation",
                "abutment",
                "superstructure",
                "top_surface",
                "railing",
                "traffic_sign",
                "pillar",
            ]
        )

    def __getitem__(self, idx):
        idx_to_use = idx % len(self.sub_clouds)
        sub_cloud = deepcopy(self.sub_clouds[idx_to_use])
        scan = sub_cloud.pts
        color = sub_cloud.color
        color /= 2**8

        if sub_cloud.labels is not None:
            label = sub_cloud.labels
            label = label.astype(np.int64)
        else:
            label = None

        # center
        scan -= sub_cloud.pos

        sample = {
            "coords": scan,
            "feats": color,
            "labels": label,
            "pos": sub_cloud.pos,
            "file_name": sub_cloud.file,
        }

        if self.apply_transform:
            sample = self.transform(sample)

        return sample
