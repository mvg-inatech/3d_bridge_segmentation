"""Originally from
https://github.com/Pointcept/PointTransformerV2/blob/5386c4d71f3d6c42c24a8105fce8750e9355dc54/pcr/datasets/transform.py
"""

import numpy as np
import torch


def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(
        arr.shape[0], dtype=np.uint64
    )
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def ravel_hash_vec(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert arr.ndim == 2
    arr = arr.copy()
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


def ravel_hash_vec_torch(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert arr.ndim == 2
    arr = arr.clone()
    arr -= arr.min(0)[0]
    arr = arr.long()
    arr_max = arr.max(0)[0].long() + 1

    keys = torch.zeros(arr.shape[0], device=arr.device).long()
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


def voxelize(coords, voxel_size=0.05, hash_type="fnv", mode=0):
    discrete_coords = np.floor(
        (coords + abs(np.min(coords, axis=0))) / np.array(voxel_size)
    )
    if hash_type == "ravel":
        key = ravel_hash_vec(discrete_coords)
    else:
        key = fnv_hash_vec(discrete_coords)

    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, count = np.unique(key_sort, return_counts=True)
    if mode == 0:  # train mode
        idx_select = (
            np.cumsum(np.insert(count, 0, 0)[0:-1])
            + np.random.randint(0, count.max(), count.size) % count
        )
        idx_unique = idx_sort[idx_select]
    return idx_unique, discrete_coords[idx_unique]


def voxelize_torch(coords, voxel_size=0.05, mode=0):
    discrete_coords = torch.floor(
        (coords + abs(torch.min(coords, dim=0)[0])) / torch.tensor(voxel_size)
    )
    key = ravel_hash_vec_torch(discrete_coords)

    idx_sort = torch.argsort(key)
    key_sort = key[idx_sort]
    _, count = torch.unique(key_sort, return_counts=True)
    if mode == 0:  # train mode
        cumsum_counts = torch.cumsum(
            torch.cat((torch.tensor([0], device=coords.device), count[:-1])),
            dim=0,
        )
        max_count = torch.max(count).item()

        # Random selection within the unique keys
        random_offsets = torch.randint(0, max_count, (count.size(0),)) % count.cpu()
        idx_select = cumsum_counts + random_offsets.to(coords.device)

        idx_unique = idx_sort[idx_select]
    return idx_unique, discrete_coords[idx_unique]


def voxelize_batch(coords, offsets, voxel_size):
    batch_size = offsets.shape[0]
    new_coords = list()
    new_length = list()
    current_idx = 0
    length = 0
    for i in range(batch_size):
        idx, _ = voxelize_torch(coords[current_idx : offsets[i]], voxel_size)
        new_coords.append(coords[idx])
        length += len(idx)
        new_length.append(length)
        current_idx += offsets[i]
    new_coords = torch.concatenate(new_coords, dim=0)
    new_length = torch.IntTensor(new_length).to(coords.device)
    return new_coords, new_length
