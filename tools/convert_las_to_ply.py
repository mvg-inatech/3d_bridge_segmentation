import argparse
from os.path import join
from os import getcwd
import sys
from plyfile import PlyData, PlyElement
import numpy as np

sys.path.append(getcwd())
from common.read_point_cloud import (
    parse_dir_for_X_file,
    load_laspy,
    convert_laspy_to_np,
)


def parse_arguments(parser):
    parser.add_argument("data_dir", type=str, help="dir to current .las files")
    parser.add_argument("new_dir", type=str, help="dir to new .ply files")
    args = parser.parse_args()
    return args


def main(args):
    file_list = parse_dir_for_X_file(args.data_dir, ".las")

    for i, file in enumerate(file_list):
        las_pc = load_laspy(join(args.data_dir, file))
        pc = convert_laspy_to_np(las_pc)
        if hasattr(las_pc, "label"):
            label = las_pc.label
        else:
            raise AttributeError("File does not have labels")
        pc[:, 3:] /= 2**8
        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
        r, g, b = pc[:, 3], pc[:, 4], pc[:, 5]
        points = np.array(
            list(zip(x, y, z)), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]
        )
        color = np.array(
            list(zip(r, g, b)),
            dtype=[("red", "u1"), ("green", "u1"), ("blue", "u1")],
        )
        label = np.array(label, dtype=[("label", "u1")])

        el_1 = PlyElement.describe(points, "points")
        el_2 = PlyElement.describe(color, "color")
        el_3 = PlyElement.describe(label, "label")
        ply_pc = PlyData([el_1, el_2, el_3])
        ply_pc.write(join(args.new_dir, file.replace(".las", ".ply")))

        print("Done with file {} of {}".format(i, len(file_list)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="training",
        description="run training for specified model and data",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    args = parse_arguments(parser)
    main(args)
