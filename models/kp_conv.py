""" HIGHLY OVERLOADED
"""

import math
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Union, Tuple

import pointops
from common.voxelize import voxelize_batch

############################################################################
# Norm


class BatchNormPackMode(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.norm = nn.BatchNorm1d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(0, 1).unsqueeze(0)  # (N, C) -> (B, C, N)
        x = self.norm(x)
        x = x.squeeze(0).transpose(0, 1)  # (B, C, N) -> (N, C)
        return x

    def __repr__(self):
        return self.__class__.__name__ + "(" + self.norm.extra_repr() + ")"


class InstanceNormPackMode(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
    ):
        super().__init__()
        self.norm = nn.InstanceNorm1d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(0, 1).unsqueeze(0)  # (N, C) -> (B, C, N)
        x = self.norm(x)
        x = x.squeeze(0).transpose(0, 1)  # (B, C, N) -> (N, C)
        return x

    def __repr__(self):
        return self.__class__.__name__ + "(" + self.norm.extra_repr() + ")"


class GroupNormPackMode(nn.Module):
    def __init__(
        self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True
    ):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(0, 1).unsqueeze(0)  # (N, C) -> (B, C, N)
        x = self.norm(x)
        x = x.squeeze(0).transpose(0, 1)  # (B, C, N) -> (N, C)
        return x

    def __repr__(self):
        return self.__class__.__name__ + "(" + self.norm.extra_repr() + ")"


LayerConfig = Optional[Union[str, Dict]]


NORM_LAYERS = {
    "None": nn.Identity,
    "BatchNorm1d": nn.BatchNorm1d,
    "BatchNorm2d": nn.BatchNorm2d,
    "BatchNorm3d": nn.BatchNorm3d,
    "InstanceNorm1d": nn.InstanceNorm1d,
    "InstanceNorm2d": nn.InstanceNorm2d,
    "InstanceNorm3d": nn.InstanceNorm3d,
    "GroupNorm": nn.GroupNorm,
    "LayerNorm": nn.LayerNorm,
}


NORM_LAYERS_PACK_MODE = {
    "None": nn.Identity,
    "BatchNorm": BatchNormPackMode,
    "InstanceNorm": InstanceNormPackMode,
    "GroupNorm": GroupNormPackMode,
    "LayerNorm": nn.LayerNorm,
}


ACT_LAYERS = {
    "None": nn.Identity,
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "ELU": nn.ELU,
    "GELU": nn.GELU,
    "Sigmoid": nn.Sigmoid,
    "Softplus": nn.Softplus,
    "Tanh": nn.Tanh,
    "Identity": nn.Identity,
}


CONV_LAYERS = {
    "Linear": nn.Linear,
    "Conv1d": nn.Conv1d,
    "Conv2d": nn.Conv2d,
    "Conv3d": nn.Conv3d,
}

############################################################################
# Pool


def local_maxpool_pack_mode(feats, neighbor_indices):
    """Max pooling from neighbors in pack mode.

    Args:
        feats (Tensor): The input features in the shape of (N, C).
        neighbor_indices (LongTensor): The neighbor indices in the shape of (M, K).

    Returns:
        pooled_feats (Tensor): The pooled features in the shape of (M, C).
    """
    feats = torch.cat((feats, torch.zeros_like(feats[:1, :])), 0)  # (N+1, C)
    neighbor_feats = index_select(feats, neighbor_indices, dim=0)  # (M, K, C)
    pooled_feats = neighbor_feats.max(1)[0]  # (M, K)
    return pooled_feats


def global_avgpool_pack_mode(feats, lengths):
    """Global average pooling over batch.

    Args:
        feats (Tensor): The input features in the shape of (N, C).
        lengths (LongTensor): The length of each sample in the batch in the shape of (B).

    Returns:
        feats (Tensor): The pooled features in the shape of (B, C).
    """
    feats_list = []
    start_index = 0
    for batch_index in range(lengths.shape[0]):
        end_index = start_index + lengths[batch_index].item()
        feats_list.append(torch.mean(feats[start_index:end_index], dim=0))
        start_index = end_index
    feats = torch.stack(feats_list, dim=0)
    return feats


############################################################################
# Functions


def create_3D_rotations(axis, angle):
    """
    Create rotation matrices from a list of axes and angles. Code from wikipedia on quaternions
    :param axis: float32[N, 3]
    :param angle: float32[N,]
    :return: float32[N, 3, 3]
    """
    t1 = np.cos(angle)
    t2 = 1 - t1
    t3 = axis[:, 0] * axis[:, 0]
    t6 = t2 * axis[:, 0]
    t7 = t6 * axis[:, 1]
    t8 = np.sin(angle)
    t9 = t8 * axis[:, 2]
    t11 = t6 * axis[:, 2]
    t12 = t8 * axis[:, 1]
    t15 = axis[:, 1] * axis[:, 1]
    t19 = t2 * axis[:, 1] * axis[:, 2]
    t20 = t8 * axis[:, 0]
    t24 = axis[:, 2] * axis[:, 2]
    R = np.stack(
        [
            t1 + t2 * t3,
            t7 - t9,
            t11 + t12,
            t7 + t9,
            t1 + t2 * t15,
            t19 - t20,
            t11 - t12,
            t19 + t20,
            t1 + t2 * t24,
        ],
        axis=1,
    )

    return np.reshape(R, (-1, 3, 3))


def check_bias_from_norm_cfg(norm_cfg: Optional[Union[str, Dict]]) -> bool:
    if norm_cfg is None:
        return True
    if isinstance(norm_cfg, dict):
        norm_cfg = norm_cfg["type"]
    return not norm_cfg.startswith("BatchNorm") and not norm_cfg.startswith(
        "InstanceNorm"
    )


def index_select(inputs: torch.Tensor, indices: torch.Tensor, dim: int) -> torch.Tensor:
    """Advanced indices select.

    Returns a tensor `output` which indexes the `inputs` tensor along dimension `dim` using the entries in `indices`
    which is a `LongTensor`.

    Different from `torch.indices_select`, `indices` does not have to be 1-D. The `dim`-th dimension of `inputs` will
    be expanded to the number of dimensions in `indices`.

    For example, suppose the shape `inputs` is $(a_0, a_1, ..., a_{n-1})$, the shape of `indices` is
    $(b_0, b_1, ..., b_{m-1})$, and `dim` is $i$, then `output` is $(n+m-1)$-d tensor, whose shape is
    $(a_0, ..., a_{i-1}, b_0, b_1, ..., b_{m-1}, a_{i+1}, ..., a_{n-1})$.

    Args:
        inputs (Tensor): (a_0, a_1, ..., a_{n-1})
        indices (LongTensor): (b_0, b_1, ..., b_{m-1})
        dim (int): The dimension to index.

    Returns:
        output (Tensor): (a_0, ..., a_{dim-1}, b_0, ..., b_{m-1}, a_{dim+1}, ..., a_{n-1})
    """
    outputs = inputs.index_select(dim, indices.view(-1))

    if indices.dim() > 1:
        if dim < 0:
            dim += inputs.dim()
        output_shape = inputs.shape[:dim] + indices.shape + inputs.shape[dim + 1 :]
        outputs = outputs.view(*output_shape)

    return outputs


def spherical_Lloyd(
    radius,
    num_cells,
    dimension=3,
    fixed="center",
    approximation="monte-carlo",
    approx_n=5000,
    max_iter=500,
    momentum=0.9,
):
    """
    Creation of kernel point via Lloyd algorithm. We use an approximation of the algorithm, and compute the Voronoi
    cell centers with discretization  of space. The exact formula is not trivial with part of the sphere as sides.
    :param radius: Radius of the kernels
    :param num_cells: Number of cell (kernel points) in the Voronoi diagram.
    :param dimension: dimension of the space
    :param fixed: fix position of certain kernel points ('none', 'center' or 'verticals')
    :param approximation: Approximation method for Lloyd's algorithm ('discretization', 'monte-carlo')
    :param approx_n: Number of point used for approximation.
    :param max_iter: Maximum nu;ber of iteration for the algorithm.
    :param momentum: Momentum of the low pass filter smoothing kernel point positions
    :return: points [num_kernels, num_points, dimension]
    """
    #######################
    # Parameters definition
    #######################

    # Radius used for optimization (points are rescaled afterwards)
    radius0 = 1.0

    #######################
    # Kernel initialization
    #######################

    # Random kernel points (Uniform distribution in a sphere)
    kernel_points = np.zeros((0, dimension))
    while kernel_points.shape[0] < num_cells:
        new_points = np.random.rand(num_cells, dimension) * 2 * radius0 - radius0
        kernel_points = np.vstack((kernel_points, new_points))
        d2 = np.sum(np.power(kernel_points, 2), axis=1)
        kernel_points = kernel_points[
            np.logical_and(d2 < radius0**2, (0.9 * radius0) ** 2 < d2), :
        ]
    kernel_points = kernel_points[:num_cells, :].reshape((num_cells, -1))

    # Optional fixing
    if fixed == "center":
        kernel_points[0, :] *= 0
    if fixed == "verticals":
        kernel_points[:3, :] *= 0
        kernel_points[1, -1] += 2 * radius0 / 3
        kernel_points[2, -1] -= 2 * radius0 / 3

    ##############################
    # Approximation initialization
    ##############################

    # Initialize discretization in this method is chosen
    if approximation == "discretization":
        side_n = int(np.floor(approx_n ** (1.0 / dimension)))
        dl = 2 * radius0 / side_n
        coords = np.arange(-radius0 + dl / 2, radius0, dl)
        if dimension == 2:
            x, y = np.meshgrid(coords, coords)
            X = np.vstack((np.ravel(x), np.ravel(y))).T
        elif dimension == 3:
            x, y, z = np.meshgrid(coords, coords, coords)
            X = np.vstack((np.ravel(x), np.ravel(y), np.ravel(z))).T
        elif dimension == 4:
            x, y, z, t = np.meshgrid(coords, coords, coords, coords)
            X = np.vstack((np.ravel(x), np.ravel(y), np.ravel(z), np.ravel(t))).T
        else:
            raise ValueError("Unsupported dimension (max is 4)")
    elif approximation == "monte-carlo":
        X = np.zeros((0, dimension))
    else:
        raise ValueError(
            'Wrong approximation method chosen: "{:s}"'.format(approximation)
        )

    # Only points inside the sphere are used
    d2 = np.sum(np.power(X, 2), axis=1)
    X = X[d2 < radius0 * radius0, :]

    #####################
    # Kernel optimization
    #####################

    # Warning if at least one kernel point has no cell
    warning = False

    # moving vectors of kernel points saved to detect convergence
    max_moves = np.zeros((0,))

    for iter in range(max_iter):
        # In the case of monte-carlo, renew the sampled points
        if approximation == "monte-carlo":
            X = np.random.rand(approx_n, dimension) * 2 * radius0 - radius0
            d2 = np.sum(np.power(X, 2), axis=1)
            X = X[d2 < radius0 * radius0, :]

        # Get the distances matrix [n_approx, K, dim]
        differences = np.expand_dims(X, 1) - kernel_points
        sq_distances = np.sum(np.square(differences), axis=2)

        # Compute cell centers
        cell_inds = np.argmin(sq_distances, axis=1)
        centers = []
        for c in range(num_cells):
            bool_c = cell_inds == c
            num_c = np.sum(bool_c.astype(np.int32))
            if num_c > 0:
                centers.append(np.sum(X[bool_c, :], axis=0) / num_c)
            else:
                warning = True
                centers.append(kernel_points[c])

        # Update kernel points with low pass filter to smooth mote carlo
        centers = np.vstack(centers)
        moves = (1 - momentum) * (centers - kernel_points)
        kernel_points += moves

        # Check moves for convergence
        max_moves = np.append(max_moves, np.max(np.linalg.norm(moves, axis=1)))

        # Optional fixing
        if fixed == "center":
            kernel_points[0, :] *= 0
        if fixed == "verticals":
            kernel_points[0, :] *= 0
            kernel_points[:3, :-1] *= 0

    # Rescale kernels with real radius
    return kernel_points * radius


def kernel_point_optimization_debug(
    radius,
    num_points,
    num_kernels=1,
    dimension=3,
    fixed="center",
    ratio=0.66,
):
    """
    Creation of kernel point via optimization of potentials.
    :param radius: Radius of the kernels
    :param num_points: points composing kernels
    :param num_kernels: number of wanted kernels
    :param dimension: dimension of the space
    :param fixed: fix position of certain kernel points ('none', 'center' or 'verticals')
    :param ratio: ratio of the radius where you want the kernels points to be placed
    :return: points [num_kernels, num_points, dimension]
    """

    #######################
    # Parameters definition
    #######################

    # Radius used for optimization (points are rescaled afterwards)
    radius0 = 1
    diameter0 = 2

    # Factor multiplicating gradients for moving points (~learning rate)
    moving_factor = 1e-2
    continuous_moving_decay = 0.9995

    # Gradient threshold to stop optimization
    thresh = 1e-5

    # Gradient clipping value
    clip = 0.05 * radius0

    #######################
    # Kernel initialization
    #######################

    # Random kernel points
    kernel_points = (
        np.random.rand(num_kernels * num_points - 1, dimension) * diameter0 - radius0
    )
    while kernel_points.shape[0] < num_kernels * num_points:
        new_points = (
            np.random.rand(num_kernels * num_points - 1, dimension) * diameter0
            - radius0
        )
        kernel_points = np.vstack((kernel_points, new_points))
        d2 = np.sum(np.power(kernel_points, 2), axis=1)
        kernel_points = kernel_points[d2 < 0.5 * radius0 * radius0, :]
    kernel_points = kernel_points[: num_kernels * num_points, :].reshape(
        (num_kernels, num_points, -1)
    )

    # Optional fixing
    if fixed == "center":
        kernel_points[:, 0, :] *= 0
    if fixed == "verticals":
        kernel_points[:, :3, :] *= 0
        kernel_points[:, 1, -1] += 2 * radius0 / 3
        kernel_points[:, 2, -1] -= 2 * radius0 / 3

    #####################
    # Kernel optimization
    #####################

    saved_gradient_norms = np.zeros((10000, num_kernels))
    old_gradient_norms = np.zeros((num_kernels, num_points))
    for iter in range(10000):
        # Compute gradients
        # *****************

        # Derivative of the sum of potentials of all points
        A = np.expand_dims(kernel_points, axis=2)
        B = np.expand_dims(kernel_points, axis=1)
        interd2 = np.sum(np.power(A - B, 2), axis=-1)
        inter_grads = (A - B) / (np.power(np.expand_dims(interd2, -1), 3 / 2) + 1e-6)
        inter_grads = np.sum(inter_grads, axis=1)

        # Derivative of the radius potential
        circle_grads = 10 * kernel_points

        # All gradients
        gradients = inter_grads + circle_grads

        if fixed == "verticals":
            gradients[:, 1:3, :-1] = 0

        # Stop condition
        # **************

        # Compute norm of gradients
        gradients_norms = np.sqrt(np.sum(np.power(gradients, 2), axis=-1))
        saved_gradient_norms[iter, :] = np.max(gradients_norms, axis=1)

        # Stop if all moving points are gradients fixed (low gradients diff)

        if (
            fixed == "center"
            and np.max(np.abs(old_gradient_norms[:, 1:] - gradients_norms[:, 1:]))
            < thresh
        ):
            break
        elif (
            fixed == "verticals"
            and np.max(np.abs(old_gradient_norms[:, 3:] - gradients_norms[:, 3:]))
            < thresh
        ):
            break
        elif np.max(np.abs(old_gradient_norms - gradients_norms)) < thresh:
            break
        old_gradient_norms = gradients_norms

        # Move points
        # ***********

        # Clip gradient to get moving dists
        moving_dists = np.minimum(moving_factor * gradients_norms, clip)

        # Fix central point
        if fixed == "center":
            moving_dists[:, 0] = 0
        if fixed == "verticals":
            moving_dists[:, 0] = 0

        # Move points
        kernel_points -= (
            np.expand_dims(moving_dists, -1)
            * gradients
            / np.expand_dims(gradients_norms + 1e-6, -1)
        )

        # moving factor decay
        moving_factor *= continuous_moving_decay

    # Rescale radius to fit the wanted ratio of radius
    r = np.sqrt(np.sum(np.power(kernel_points, 2), axis=-1))
    kernel_points *= ratio / np.mean(r[:, 1:])

    # Rescale kernels with real radius
    return kernel_points * radius, saved_gradient_norms


def load_kernels(radius, num_kpoints, dimension, fixed, lloyd=False):
    # To many points switch to Lloyds
    if num_kpoints > 30:
        lloyd = True

    if lloyd:
        # Create kernels
        kernel_points = spherical_Lloyd(
            1.0, num_kpoints, dimension=dimension, fixed=fixed
        )
    else:
        # Create kernels
        kernel_points, grad_norms = kernel_point_optimization_debug(
            1.0,
            num_kpoints,
            num_kernels=100,
            dimension=dimension,
            fixed=fixed,
        )
        # Find best candidate
        best_k = np.argmin(grad_norms[-1, :])

        # Save points
        kernel_points = kernel_points[best_k, :, :]

    # Random roations for the kernel
    # N.B. 4D random rotations not supported yet
    R = np.eye(dimension)
    theta = np.random.rand() * 2 * np.pi
    if dimension == 2:
        if fixed != "vertical":
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)
    elif dimension == 3:
        if fixed != "vertical":
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        else:
            phi = (np.random.rand() - 0.5) * np.pi
            # Create the first vector in carthesian coordinates
            u = np.array(
                [np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)]
            )
            # Choose a random rotation angle
            alpha = np.random.rand() * 2 * np.pi
            # Create the rotation matrix with this vector and angle
            R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[
                0
            ]
            R = R.astype(np.float32)

    # Add a small noise
    kernel_points = kernel_points + np.random.normal(
        scale=0.01, size=kernel_points.shape
    )
    # Scale kernels
    kernel_points = radius * kernel_points
    # Rotate kernels
    kernel_points = np.matmul(kernel_points, R)

    return kernel_points.astype(np.float32)


class KPConv(nn.Module):
    """Rigid KPConv.

    Paper: https://arxiv.org/abs/1904.08889.

    Args:
         in_channels (int): The number of the input channels.
         out_channels (int): The number of the output channels.
         kernel_size (int): The number of kernel points.
         radius (float): The radius used for kernel point init.
         sigma (float): The influence radius of each kernel point.
         bias (bool, optional): If True, use bias. Default: False.
         dimension (int, optional): The dimension of the point space. Default: 3.
         inf (float, optional): The value of infinity to generate the padding point. Default: 1e6.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        radius: float,
        sigma: float,
        groups: int = 1,
        bias: bool = False,
        dimension: int = 3,
        inf: float = 1e6,
    ):
        """Initialize a rigid KPConv."""
        super().__init__()

        assert in_channels % groups == 0, "in_channels must be divisible by groups."
        assert out_channels % groups == 0, "out_channels must be divisible by groups."
        in_channels_per_group = in_channels // groups
        out_channels_per_group = out_channels // groups

        # Save parameters
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.sigma = sigma
        self.groups = groups
        self.dimension = dimension
        self.inf = inf
        self.in_channels_per_group = in_channels_per_group
        self.out_channels_per_group = out_channels_per_group

        # Initialize weights
        if self.groups == 1:
            weights = torch.zeros(size=(kernel_size, in_channels, out_channels))
        else:
            weights = torch.zeros(
                size=(
                    kernel_size,
                    groups,
                    in_channels_per_group,
                    out_channels_per_group,
                )
            )
        self.weights = nn.Parameter(weights)

        if bias:
            self.bias = nn.Parameter(torch.zeros(size=(out_channels,)))
        else:
            self.register_parameter("bias", None)

        # Reset parameters
        self.reset_parameters()

        # Initialize kernel points
        kernel_points = self.initialize_kernel_points()  # (N, 3)
        self.register_buffer("kernel_points", kernel_points)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def initialize_kernel_points(self) -> torch.Tensor:
        """Initialize the kernel point positions in a sphere."""
        kernel_points = load_kernels(
            self.radius, self.kernel_size, dimension=self.dimension, fixed="center"
        )
        return torch.from_numpy(kernel_points).float()

    def forward(
        self,
        q_points: torch.Tensor,
        s_points: torch.Tensor,
        s_feats: torch.Tensor,
        neighbor_indices: torch.Tensor,
    ) -> torch.Tensor:
        """KPConv forward.

        Args:
            s_feats (Tensor): (N, C_in)
            q_points (Tensor): (M, 3)
            s_points (Tensor): (N, 3)
            neighbor_indices (LongTensor): (M, H)

        Returns:
            q_feats (Tensor): (M, C_out)
        """
        padded_s_points = torch.cat(
            [s_points, torch.zeros_like(s_points[:1, :]) + self.inf], 0
        )  # (N, 3) -> (N+1, 3)
        neighbors = index_select(
            padded_s_points, neighbor_indices, dim=0
        )  # (N+1, 3) -> (M, H, 3)
        neighbors = neighbors - q_points.unsqueeze(1)  # (M, H, 3)

        # Get Kernel point influences
        neighbors = neighbors.unsqueeze(2)  # (M, H, 3) -> (M, H, 1, 3)
        differences = (
            neighbors - self.kernel_points
        )  # (M, H, 1, 3) x (K, 3) -> (M, H, K, 3)
        sq_distances = torch.sum(differences**2, dim=3)  # (M, H, K)
        neighbor_weights = torch.clamp(
            1 - torch.sqrt(sq_distances) / self.sigma, min=0.0
        )  # (M, H, K)
        neighbor_weights = torch.transpose(
            neighbor_weights, 1, 2
        )  # (M, H, K) -> (M, K, H)

        # apply neighbor weights
        padded_s_feats = torch.cat(
            (s_feats, torch.zeros_like(s_feats[:1, :])), 0
        )  # (N, C) -> (N+1, C)
        neighbor_feats = index_select(
            padded_s_feats, neighbor_indices, dim=0
        )  # (N+1, C) -> (M, H, C)
        weighted_feats = torch.matmul(
            neighbor_weights, neighbor_feats
        )  # (M, K, H) x (M, H, C) -> (M, K, C)

        # apply convolutional weights
        if self.groups == 1:
            # standard conv
            output_feats = torch.einsum("mkc,kcd->md", weighted_feats, self.weights)
        else:
            # group conv
            weighted_feats = weighted_feats.view(
                -1, self.kernel_size, self.groups, self.in_channels_per_group
            )
            output_feats = torch.einsum("mkgc,kgcd->mgd", weighted_feats, self.weights)
            output_feats = output_feats.view(-1, self.out_channels)

        # density normalization
        neighbor_feats_sum = torch.sum(neighbor_feats, dim=-1)  # (M, H)
        neighbor_num = torch.sum(torch.gt(neighbor_feats_sum, 0.0), dim=-1)  # (M,)
        neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))  # (M,)
        output_feats = output_feats / neighbor_num.unsqueeze(1)

        # NOTE: normalization with only positive neighbors works slightly better than all neighbors
        # neighbor_num = torch.sum(torch.lt(neighbor_indices, s_points.shape[0]), dim=-1)  # (M,)
        # neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))  # (M,)
        # output_feats = output_feats / neighbor_num.unsqueeze(1)

        # add bias
        if self.bias is not None:
            output_feats = output_feats + self.bias

        return output_feats

    def extra_repr(self) -> str:
        param_strings = [
            f"kernel_size={self.kernel_size}",
            f"in_channels={self.in_channels}",
            f"out_channels={self.out_channels}",
            f"radius={self.radius:g}",
            f"sigma={self.sigma:g}",
            f"bias={self.bias is not None}",
        ]
        if self.dimension != 3:
            param_strings.append(f"dimension={self.dimension}")
        format_string = ", ".join(param_strings)
        return format_string


############################################################################
# Builder


def parse_cfg(cfg: LayerConfig) -> Tuple[str, Dict]:
    if cfg is None:
        return "None", {}
    if isinstance(cfg, str):
        return cfg, {}
    assert isinstance(cfg, Dict), "Illegal cfg type: {}.".format(type(cfg))
    layer = cfg["type"]
    kwargs = {key: value for key, value in cfg.items() if key != "type"}
    return layer, kwargs


def _find_optimal_num_groups(num_channels: int) -> int:
    """Find the optimal number of groups for GroupNorm."""
    # strategy: at most 32 groups, at least 8 channels per group
    num_groups = 32
    while num_groups > 1:
        if num_channels % num_groups == 0:
            num_channels_per_group = num_channels // num_groups
            if num_channels_per_group >= 8:
                break
        num_groups = num_groups // 2
    assert num_groups != 1, (
        f"Cannot find 'num_groups' in GroupNorm with 'num_channels={num_channels}' automatically. "
        "Please manually specify 'num_groups'."
    )
    return num_groups


def _configure_norm_args(layer: str, kwargs: Dict, num_features: int) -> Dict:
    """Configure norm args."""
    if layer == "GroupNorm":
        kwargs["num_channels"] = num_features
        if "num_groups" not in kwargs:
            kwargs["num_groups"] = _find_optimal_num_groups(num_features)
    elif layer == "LayerNorm":
        kwargs["normalized_shape"] = num_features
    elif layer != "None":
        kwargs["num_features"] = num_features
    return kwargs


def _configure_act_args(layer: str, kwargs: Dict) -> Dict:
    """Configure activation args."""
    if layer == "LeakyReLU":
        if "negative_slope" not in kwargs:
            kwargs["negative_slope"] = 0.2
    return kwargs


def build_norm_layer(num_features, norm_cfg: LayerConfig) -> nn.Module:
    """Factory function for normalization layers."""
    layer, kwargs = parse_cfg(norm_cfg)
    assert layer in NORM_LAYERS, f"Illegal normalization: {layer}."
    kwargs = _configure_norm_args(layer, kwargs, num_features)
    return NORM_LAYERS[layer](**kwargs)


def build_norm_layer_pack_mode(num_features, norm_cfg: LayerConfig) -> nn.Module:
    """Factory function for normalization layers in pack mode."""
    layer, kwargs = parse_cfg(norm_cfg)
    assert layer in NORM_LAYERS_PACK_MODE, f"Illegal normalization: {layer}."
    kwargs = _configure_norm_args(layer, kwargs, num_features)
    return NORM_LAYERS_PACK_MODE[layer](**kwargs)


def build_act_layer(act_cfg: LayerConfig) -> nn.Module:
    """Factory function for activation functions."""
    layer, kwargs = parse_cfg(act_cfg)
    assert layer in ACT_LAYERS, f"Illegal activation: {layer}."
    kwargs = _configure_act_args(layer, kwargs)
    return ACT_LAYERS[layer](**kwargs)


def build_conv_layer(conv_cfg: Dict) -> nn.Module:
    """Factory function for convolution or linear layers."""
    layer, kwargs = parse_cfg(conv_cfg)
    assert layer in CONV_LAYERS, f"Illegal conv layer: {layer}."
    return CONV_LAYERS[layer](**kwargs)


############################################################################
# Blocks


class KPConvBlock(nn.Module):
    """KPConv block with normalization and activation.

    Args:
        in_channels (int): dimension input features
        out_channels (int): dimension input features
        kernel_size (int): number of kernel points
        radius (float): convolution radius
        sigma (float): influence radius of kernel points
        dimension (int=3): dimension of input
        norm_cfg (str|dict|None='GroupNorm'): normalization config
        act_cfg (str|dict|None='LeakyReLU'): activation config
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        radius: float,
        sigma: float,
        groups: int = 1,
        dimension: int = 3,
        norm_cfg: LayerConfig = "GroupNorm",
        act_cfg: LayerConfig = "LeakyReLU",
    ):
        super().__init__()

        bias = check_bias_from_norm_cfg(norm_cfg)
        self.conv = KPConv(
            in_channels,
            out_channels,
            kernel_size,
            radius,
            sigma,
            groups=groups,
            bias=bias,
            dimension=dimension,
        )

        self.norm = build_norm_layer_pack_mode(out_channels, norm_cfg)
        self.act = build_act_layer(act_cfg)

    def forward(self, q_points, s_points, s_feats, neighbor_indices):
        q_feats = self.conv(q_points, s_points, s_feats, neighbor_indices)
        q_feats = self.norm(q_feats)
        q_feats = self.act(q_feats)
        return q_feats


class KPResidualBlock(nn.Module):
    """KPConv residual bottleneck block.

    Args:
        in_channels (int): dimension input features
        out_channels (int): dimension input features
        kernel_size (int): number of kernel points
        radius (float): convolution radius
        sigma (float): influence radius of each kernel point
        dimension (int=3): dimension of input
        strided (bool=False): strided or not
        norm_cfg (str|dict|None='GroupNorm'): normalization config
        act_cfg (str|dict|None='LeakyReLU'): activation config
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        radius: float,
        sigma: float,
        groups: int = 1,
        dimension: int = 3,
        strided: bool = False,
        norm_cfg: LayerConfig = "GroupNorm",
        act_cfg: LayerConfig = "LeakyReLU",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strided = strided

        mid_channels = out_channels // 4

        self.unary1 = UnaryBlockPackMode(
            in_channels, mid_channels, norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.conv = KPConvBlock(
            mid_channels,
            mid_channels,
            kernel_size,
            radius,
            sigma,
            groups=groups,
            dimension=dimension,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.unary2 = UnaryBlockPackMode(
            mid_channels, out_channels, norm_cfg=norm_cfg, act_cfg=None
        )

        if in_channels != out_channels:
            self.unary_shortcut = UnaryBlockPackMode(
                in_channels, out_channels, norm_cfg=norm_cfg, act_cfg=None
            )
        else:
            self.unary_shortcut = nn.Identity()

        self.act = build_act_layer(act_cfg)

    def forward(self, q_points, s_points, s_feats, neighbor_indices):
        residual = self.unary1(s_feats)
        residual = self.conv(q_points, s_points, residual, neighbor_indices)
        residual = self.unary2(residual)

        if self.strided:
            shortcut = local_maxpool_pack_mode(s_feats, neighbor_indices)
        else:
            shortcut = s_feats
        shortcut = self.unary_shortcut(shortcut)

        q_feats = residual + shortcut
        q_feats = self.act(q_feats)

        return q_feats


class UnaryBlockPackMode(nn.Module):
    """Unary block with normalization and activation in pack mode.

    Args:
        in_channels (int): dimension input features
        out_channels (int): dimension input features
        norm_cfg (str|dict|None='GroupNorm'): normalization config
        act_cfg (str|dict|None='LeakyReLU'): activation config
    """

    def __init__(
        self, in_channels, out_channels, norm_cfg="GroupNorm", act_cfg="LeakyReLU"
    ):
        super().__init__()

        bias = check_bias_from_norm_cfg(norm_cfg)
        self.mlp = nn.Linear(in_channels, out_channels, bias=bias)

        self.norm = build_norm_layer_pack_mode(out_channels, norm_cfg)
        self.act = build_act_layer(act_cfg)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        feats = self.mlp(feats)
        feats = self.norm(feats)
        feats = self.act(feats)
        return feats


############################################################################
# Architecture


class KPFCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.num_stages = cfg.num_stages
        self.voxel_size = cfg.basic_voxel_size
        self.kpconv_radius = cfg.kpconv_radius
        self.kpconv_sigma = cfg.kpconv_sigma
        self.neighbor_limits = cfg.neighbor_limits
        self.first_radius = self.voxel_size * self.kpconv_radius
        self.first_sigma = self.voxel_size * self.kpconv_sigma

        input_dim = cfg.in_channels
        first_dim = cfg.init_dim
        kernel_size = cfg.kernel_size
        first_radius = self.first_radius
        first_sigma = self.first_sigma

        self.encoder1_1 = KPConvBlock(
            input_dim, first_dim, kernel_size, first_radius, first_sigma
        )
        self.encoder1_2 = KPResidualBlock(
            first_dim, first_dim * 2, kernel_size, first_radius, first_sigma
        )

        self.encoder2_1 = KPResidualBlock(
            first_dim * 2,
            first_dim * 2,
            kernel_size,
            first_radius,
            first_sigma,
            strided=True,
        )
        self.encoder2_2 = KPResidualBlock(
            first_dim * 2,
            first_dim * 4,
            kernel_size,
            first_radius * 2,
            first_sigma * 2,
        )
        self.encoder2_3 = KPResidualBlock(
            first_dim * 4,
            first_dim * 4,
            kernel_size,
            first_radius * 2,
            first_sigma * 2,
        )

        self.encoder3_1 = KPResidualBlock(
            first_dim * 4,
            first_dim * 4,
            kernel_size,
            first_radius * 2,
            first_sigma * 2,
            strided=True,
        )
        self.encoder3_2 = KPResidualBlock(
            first_dim * 4,
            first_dim * 8,
            kernel_size,
            first_radius * 4,
            first_sigma * 4,
        )
        self.encoder3_3 = KPResidualBlock(
            first_dim * 8,
            first_dim * 8,
            kernel_size,
            first_radius * 4,
            first_sigma * 4,
        )

        self.encoder4_1 = KPResidualBlock(
            first_dim * 8,
            first_dim * 8,
            kernel_size,
            first_radius * 4,
            first_sigma * 4,
            strided=True,
        )
        self.encoder4_2 = KPResidualBlock(
            first_dim * 8,
            first_dim * 16,
            kernel_size,
            first_radius * 8,
            first_sigma * 8,
        )
        self.encoder4_3 = KPResidualBlock(
            first_dim * 16,
            first_dim * 16,
            kernel_size,
            first_radius * 8,
            first_sigma * 8,
        )

        self.encoder5_1 = KPResidualBlock(
            first_dim * 16,
            first_dim * 16,
            kernel_size,
            first_radius * 8,
            first_sigma * 8,
            strided=True,
        )
        self.encoder5_2 = KPResidualBlock(
            first_dim * 16,
            first_dim * 32,
            kernel_size,
            first_radius * 16,
            first_sigma * 16,
        )
        self.encoder5_3 = KPResidualBlock(
            first_dim * 32,
            first_dim * 32,
            kernel_size,
            first_radius * 16,
            first_sigma * 16,
        )

        self.decoder4 = UnaryBlockPackMode(first_dim * 48, first_dim * 16)
        self.decoder3 = UnaryBlockPackMode(first_dim * 24, first_dim * 8)
        self.decoder2 = UnaryBlockPackMode(first_dim * 12, first_dim * 4)
        self.decoder1 = UnaryBlockPackMode(first_dim * 6, first_dim * 2)

        self.classifier = nn.Sequential(
            nn.Linear(first_dim * 2, first_dim),
            nn.GroupNorm(8, first_dim),
            nn.ReLU(),
            nn.Linear(first_dim, cfg.num_classes),
        )

    def forward(self, data_dict):
        output_dict = {}

        feats_0 = data_dict["feats"]
        points_0 = data_dict["coords"]
        offsets_0 = data_dict["offsets"]

        ####################################
        # Encode

        neighbors_0 = pointops.ball_query(
            self.neighbor_limits[0],
            self.first_radius,
            0.0,
            points_0,
            offsets_0,
        )[0]
        # set not existing neighbors to the last point (which will be added with 1e6 later)
        neighbors_0[torch.where(neighbors_0 == -1)] = points_0.shape[0]
        feats_s1 = torch.cat([torch.ones_like(feats_0[:, :1]), feats_0], dim=1)

        feats_s1 = self.encoder1_1(points_0, points_0, feats_s1, neighbors_0)
        feats_s1 = self.encoder1_2(points_0, points_0, feats_s1, neighbors_0)

        ####################################
        # Encode down 1

        points_1, offsets_1 = voxelize_batch(points_0, offsets_0, self.voxel_size * 2)
        subsample_neighbors_0 = pointops.ball_query(
            self.neighbor_limits[0],
            self.first_radius,
            0.0,
            points_0,
            offsets_0,
            points_1,
            offsets_1,
        )[0]
        subsample_neighbors_0[torch.where(subsample_neighbors_0 == -1)] = (
            points_1.shape[0]
        )
        neighbors_1 = pointops.ball_query(
            self.neighbor_limits[0],
            self.first_radius * 2,
            0.0,
            points_1,
            offsets_1,
        )[0]
        neighbors_1[torch.where(neighbors_1 == -1)] = points_1.shape[0]

        feats_s2 = self.encoder2_1(points_1, points_0, feats_s1, subsample_neighbors_0)
        feats_s2 = self.encoder2_2(points_1, points_1, feats_s2, neighbors_1)
        feats_s2 = self.encoder2_3(points_1, points_1, feats_s2, neighbors_1)

        ####################################
        # Encode down 2

        points_2, offsets_2 = voxelize_batch(
            points_1, offsets_1, self.voxel_size * 2 * 2
        )
        subsample_neighbors_1 = pointops.ball_query(
            self.neighbor_limits[1],
            self.first_radius * 2 * 2,
            0.0,
            points_1,
            offsets_1,
            points_2,
            offsets_2,
        )[0]
        subsample_neighbors_1[torch.where(subsample_neighbors_1 == -1)] = (
            points_2.shape[0]
        )
        neighbors_2 = pointops.ball_query(
            self.neighbor_limits[1],
            self.first_radius * 2 * 2,
            0.0,
            points_2,
            offsets_2,
        )[0]
        neighbors_2[torch.where(neighbors_2 == -1)] = points_2.shape[0]

        feats_s3 = self.encoder3_1(points_2, points_1, feats_s2, subsample_neighbors_1)
        feats_s3 = self.encoder3_2(points_2, points_2, feats_s3, neighbors_2)
        feats_s3 = self.encoder3_3(points_2, points_2, feats_s3, neighbors_2)

        ####################################
        # Encode down 3

        points_3, offsets_3 = voxelize_batch(
            points_2, offsets_2, self.voxel_size * 2 * 2 * 2
        )
        subsample_neighbors_2 = pointops.ball_query(
            self.neighbor_limits[2],
            self.first_radius * 2 * 2 * 2,
            0.0,
            points_2,
            offsets_2,
            points_3,
            offsets_3,
        )[0]
        subsample_neighbors_2[torch.where(subsample_neighbors_2 == -1)] = (
            points_3.shape[0]
        )
        neighbors_3 = pointops.ball_query(
            self.neighbor_limits[2],
            self.first_radius * 2 * 2 * 2,
            0.0,
            points_3,
            offsets_3,
        )[0]
        neighbors_3[torch.where(neighbors_3 == -1)] = points_3.shape[0]

        feats_s4 = self.encoder4_1(points_3, points_2, feats_s3, subsample_neighbors_2)
        feats_s4 = self.encoder4_2(points_3, points_3, feats_s4, neighbors_3)
        feats_s4 = self.encoder4_3(points_3, points_3, feats_s4, neighbors_3)

        ####################################
        # Encode down 4

        points_4, offsets_4 = voxelize_batch(
            points_3, offsets_3, self.voxel_size * 2 * 2 * 2 * 2
        )
        subsample_neighbors_3 = pointops.ball_query(
            self.neighbor_limits[3],
            self.first_radius * 2 * 2 * 2 * 2,
            0.0,
            points_3,
            offsets_3,
            points_4,
            offsets_4,
        )[0]
        subsample_neighbors_3[torch.where(subsample_neighbors_3 == -1)] = (
            points_4.shape[0]
        )
        neighbors_4 = pointops.ball_query(
            self.neighbor_limits[3],
            self.first_radius * 2 * 2 * 2 * 2,
            0.0,
            points_4,
            offsets_4,
        )[0]
        neighbors_4[torch.where(neighbors_4 == -1)] = points_4.shape[0]

        feats_s5 = self.encoder5_1(points_4, points_3, feats_s4, subsample_neighbors_3)
        feats_s5 = self.encoder5_2(points_4, points_4, feats_s5, neighbors_4)
        feats_s5 = self.encoder5_3(points_4, points_4, feats_s5, neighbors_4)

        latent_s5 = feats_s5

        latent_s4 = pointops.interpolation(
            points_4.contiguous(),
            points_3.contiguous(),
            latent_s5.contiguous(),
            offsets_4,
            offsets_3,
        )
        latent_s4 = torch.cat([latent_s4, feats_s4], dim=1)
        latent_s4 = self.decoder4(latent_s4)

        latent_s3 = pointops.interpolation(
            points_3.contiguous(),
            points_2.contiguous(),
            latent_s4.contiguous(),
            offsets_3,
            offsets_2,
        )
        latent_s3 = torch.cat([latent_s3, feats_s3], dim=1)
        latent_s3 = self.decoder3(latent_s3)

        latent_s2 = pointops.interpolation(
            points_2.contiguous(),
            points_1.contiguous(),
            latent_s3.contiguous(),
            offsets_2,
            offsets_1,
        )
        latent_s2 = torch.cat([latent_s2, feats_s2], dim=1)
        latent_s2 = self.decoder2(latent_s2)

        latent_s1 = pointops.interpolation(
            points_1.contiguous(),
            points_0.contiguous(),
            latent_s2.contiguous(),
            offsets_1,
            offsets_0,
        )
        latent_s1 = torch.cat([latent_s1, feats_s1], dim=1)
        latent_s1 = self.decoder1(latent_s1)

        scores = self.classifier(latent_s1)

        return scores
