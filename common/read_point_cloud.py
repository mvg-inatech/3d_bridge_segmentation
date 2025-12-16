import os
from os import listdir
import logging
import laspy
import numpy as np
from plyfile import PlyData

############################################################################
# PARSING


def parse_dir_for_X_file(dir: str, X_ending: str):
    wanted_files = []
    files = listdir(dir)
    for f in files:
        if f.endswith(X_ending):
            wanted_files.append(f)
    logging.info(
        "Found {} {} files in dir: {}".format(len(wanted_files), X_ending, dir)
    )
    return wanted_files


############################################################################
# OPENING


def load_laspy(dir: str) -> laspy.lasdata.LasData:
    with laspy.open(dir) as fh:
        las = fh.read()
    return las


def read_laspy_data(dir: str):
    data = load_laspy(dir)
    pts = np.zeros((len(data), 3))
    pts[:, 0] = data.x
    pts[:, 1] = data.y
    pts[:, 2] = data.z
    color = np.zeros((len(data), 3))
    color[:, 0] = data.red
    color[:, 1] = data.green
    color[:, 2] = data.blue
    if hasattr(data, "label"):
        label = data.label.astype(np.int64)
    else:
        label = None
    return pts, color, label


def load_ply(dir: str) -> PlyData:
    with open(dir, "rb") as quad_file:
        plydata = PlyData.read(quad_file)
    return plydata


def read_ply_data(dir: str):
    plydata = load_ply(dir)
    pts = np.zeros((len(plydata.elements[0].data["x"]), 3))
    pts[:, 0] = plydata.elements[0].data["x"]
    pts[:, 1] = plydata.elements[0].data["y"]
    pts[:, 2] = plydata.elements[0].data["z"]
    color = np.zeros((len(pts), 3))
    color[:, 0] = plydata.elements[1].data["red"]
    color[:, 1] = plydata.elements[1].data["green"]
    color[:, 2] = plydata.elements[1].data["blue"]
    if "label" in plydata.header:
        label = plydata.elements[2].data["label"].astype(np.int64)
    else:
        label = None
    return pts, color, label


############################################################################
# SAVING


def save_laspy(las_pc: laspy.lasdata.LasData, name: str = "test.las") -> None:
    las_pc.write(name)
    logging.info("Point cloud saved with name {}".format(name))


def save_pred_to_laspy(pc: np.array, pred: np.array, dir: str) -> None:
    header = laspy.LasHeader(point_format=7)

    # header offset and scale have to be specified
    xmin = np.floor(np.min(pc[:, 0]))
    ymin = np.floor(np.min(pc[:, 1]))
    zmin = np.floor(np.min(pc[:, 2]))
    header.offset = [xmin, ymin, zmin]
    header.scale = [0.001, 0.001, 0.001]

    las = laspy.LasData(header)
    las.add_extra_dim(
        laspy.ExtraBytesParams(
            name="prediction", type=np.uint64, description="Class label"
        )
    )
    las.x = pc[:, 0]
    las.y = pc[:, 1]
    las.z = pc[:, 2]
    try:
        las.red = pc[:, 3]
        las.green = pc[:, 4]
        las.blue = pc[:, 5]
    except:
        logging.warning("Unable to write colorinformation")
    las.prediction = pred
    save_laspy(las, name=dir)


############################################################################
# CONVERTING


def convert_laspy_to_np(las_pc: laspy.lasdata.LasData) -> np.array:
    points = np.stack(
        (np.asarray(las_pc.x), np.asarray(las_pc.y), np.asarray(las_pc.z)), axis=1
    )
    try:
        colors = np.stack(
            (
                np.asarray(las_pc.red),
                np.asarray(las_pc.green),
                np.asarray(las_pc.blue),
            ),
            axis=1,
        )
        point_cloud = np.concatenate((points, colors), axis=1)
        return point_cloud
    except:
        return points
