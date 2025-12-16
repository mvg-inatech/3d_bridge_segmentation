"""
ONLY SUPPORTED RUNNING ON CUDA -> spconv and pointsops deps

Always stores just the reduced point cloud!
"""

from os.path import join
import argparse
import torch
import numpy as np
from common.parser import get_params
from models.model_loader import get_model
from common.read_point_cloud import (
    parse_dir_for_X_file,
    read_ply_data,
    read_laspy_data,
    save_pred_to_laspy,
)
from dataloader.base_loader import calculate_sub_idx
from dataloader.loader_utils import (
    VoxelDownsamplingCollator,
    VoxelDiscreteCollator,
    sample_to_device,
)
from common.metric import IoUEval


def parse_arguments(parser):
    parser.add_argument("config_dir", type=str, help="dir to config file")
    parser.add_argument("data_dir", type=str, help="dir to data files")
    args = parser.parse_args()
    return args


def main(args):
    # model stuff
    device = torch.device("cuda")
    cfg_dir = join(args.config_dir, "config.yaml")
    config = get_params(cfg_dir)
    model = get_model(config["name"], cfg_dir)
    model_dict = torch.load(join(args.config_dir, "bird.pt"))
    model.load_state_dict(model_dict)
    model = model.to(device)
    model.eval()

    # data stuff
    las_files = parse_dir_for_X_file(args.data_dir, ".las")
    ply_files = parse_dir_for_X_file(args.data_dir, ".ply")
    file_list = las_files + ply_files
    print("Found {} file(s)".format(len(file_list)))
    if config["collator"] == "point":
        collator = VoxelDownsamplingCollator(config["voxel_size"])
    elif config["collator"] == "voxel":
        collator = VoxelDiscreteCollator(config["voxel_size"])
    else:
        raise NotImplementedError(
            "Collator {} not implemented".format(config["collator"])
        )

    iou_eval_all = IoUEval(config["model"]["num_classes"], device)
    for file in file_list:
        iou_eval_single = IoUEval(config["model"]["num_classes"], device)
        print("Starting on {}".format(file))
        if file.endswith(".las"):
            pts, color, label = read_laspy_data(join(args.data_dir, file))
        elif file.endswith(".ply"):
            pts, color, label = read_ply_data(join(args.data_dir, file))
        else:
            raise NotImplementedError(f"Ending for {file} not implemented")

        # sub indicies for going for large clouds
        pts -= np.median(pts, axis=0)
        sub_idx_list = calculate_sub_idx(pts, config["bound"])

        pred_pts = list()
        pred_color = list()
        preds = list()
        label_list = list()
        for sub_idx in sub_idx_list:
            batch = dict()
            batch["coords"] = pts[sub_idx.idx] - sub_idx.pos
            batch["feats"] = color[sub_idx.idx] / 255
            if label is not None:
                batch["labels"] = label[sub_idx.idx]
            else:
                batch["labels"] = None
            batch = collator(list([batch]))
            with torch.no_grad():
                out = model(sample_to_device(batch, device))
                pred = torch.argmax(out, dim=1)

            pred_pts.append(batch["coords"].detach().cpu().numpy() + sub_idx.pos)
            pred_color.append(batch["feats"].detach().cpu().numpy() * 255)
            preds.append(pred.detach().cpu().numpy())
            if label is not None:
                label_list.append(batch["labels"].detach().cpu().numpy())

        pred_pts = np.concatenate(pred_pts)
        pred_color = np.concatenate(pred_color)
        preds = np.concatenate(preds)
        if label is not None:
            label_red = np.concatenate(label_list)
            iou_eval_all.add_batch(preds, label_red)
            iou_eval_single.add_batch(preds, label_red)
            print(iou_eval_single.get_IoU())

        # stora as .las
        if file.endswith(".ply"):
            file = file.replace(".ply", ".las")
        save_pred_to_laspy(np.concatenate((pred_pts, pred_color), axis=1), preds, file)

        print("HUI")
    print("Final IoU score for all files:")
    print(iou_eval_all.get_IoU())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="testing",
        description="run testing for specified model and data",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    args = parse_arguments(parser)
    main(args)
