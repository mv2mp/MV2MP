import numpy as np

from scipy.spatial.transform import Rotation as R

import jaxtyping as jt
import cv2

import collections.abc
import functools
import math
import random
from typing import Callable, Literal, Sequence, Tuple, Union, overload

import nerfacc
import torch

from . import camera

def pt2np(arr: torch.Tensor) -> np.ndarray:
    return arr.detach().cpu().numpy()


@overload
def np2pt(data: np.ndarray, device: Union[torch.device, str] = "cpu") -> torch.Tensor:
    ...


@overload
def np2pt(data: Sequence[np.ndarray], device: Union[torch.device, str] = "cpu") -> Sequence[torch.Tensor]:
    ...


def np2pt(
    data: Union[np.ndarray, Sequence[np.ndarray]],
    device: Union[torch.device, str] = "cpu",
) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
    if isinstance(data, collections.abc.Sequence):
        return type(data)(np2pt(v, device=device) for v in data)  # noqa: typing

    if data.dtype == np.float64:
        data = data.astype(np.float32)
    elif data.dtype == np.uint64:
        data = data.astype(np.int64)
    elif data.dtype == np.uint32:
        data = data.astype(np.int32)
    return torch.from_numpy(data).to(device)


def optionally_deterministic(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        seed = kwargs.pop("seed", None)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        return f(*args, **kwargs)

    return wrapped


@optionally_deterministic
def get_random_coo_index(
    length: int,
    num_samples: int,
) -> jt.Int[torch.Tensor, "M"]:
    if num_samples >= length:
        return torch.arange(length)
    else:
        return torch.randperm(length)[:num_samples]


@optionally_deterministic
def get_random_coo_indexing_array(
    batch_lengths: Sequence[int],
    num_samples_per_instance: int = 64,
) -> jt.Int[torch.Tensor, "M"]:
    indices = [get_random_coo_index(batch_len, num_samples_per_instance) for batch_len in batch_lengths]

    offset_indices = []
    accumulated_length = 0
    for index_pack, batch_offset in zip(indices, batch_lengths):
        offset_indices.append(accumulated_length + index_pack)
        accumulated_length += batch_offset

    return torch.cat(offset_indices, dim=0)


def get_world_frame_translate(Rt: jt.Float[torch.Tensor, "N 3 4"]) -> jt.Float[torch.Tensor, "N 3"]:
    R, tr = torch.split(Rt, [3, 1], dim=2)
    tr = tr[..., 0]
    return torch.einsum("nij,ni->nj", R, -tr)


def Rt_inverse(Rt: jt.Float[torch.Tensor, "N 3 4"]) -> jt.Float[torch.Tensor, "N 3 4"]:
    to_cat = torch.zeros(Rt.shape[0], 1, 4, dtype=Rt.dtype, device=Rt.device)
    to_cat[:, 0, -1] = 1.0
    Rt_cat = torch.cat([Rt, to_cat], dim=1)

    inversed = torch.linalg.inv(Rt_cat)
    return inversed[:, :3]


def to_homogeneous(arr: jt.Float[torch.Tensor, "* D"]) -> jt.Float[torch.Tensor, "* D+1"]:
    cat_part = torch.ones_like(arr[..., :1], dtype=arr.dtype)
    return torch.cat([arr, cat_part], dim=-1)


def to_homogeneous_np(arr: jt.Float[np.ndarray, "* D"]) -> jt.Float[np.ndarray, "* D+1"]:
    cat_part = np.ones_like(arr[..., :1], dtype=arr.dtype)
    return np.concatenate([arr, cat_part], axis=-1)


def ravel_index(
    index: jt.Int[torch.Tensor, "*N I"],
    shape: torch.Size,
) -> jt.Int[torch.Tensor, "*N"]:
    shape_oi = shape[1:][::-1]

    shape_oi = torch.tensor((1,) + shape_oi, dtype=torch.int64, device=index.device)
    shape_oi = torch.cumprod(shape_oi, dim=0).flip(0)
    index = (index * shape_oi).sum(dim=-1)
    return index


@torch.no_grad()
def importance_sampling(
    sorted_bins: jt.Float[torch.Tensor, "n_bins 2"],
    pdf: jt.Float[torch.Tensor, "n_bins"],
    distribution_index: jt.Int[torch.Tensor, "n_bins"],
    n_samples: Union[int, float],
) -> Tuple[jt.Float[torch.Tensor, "N 2"], jt.Int[torch.Tensor, "N"]]:
    incoming_unique = torch.unique(distribution_index)
    incoming_index_range = torch.arange(0, len(incoming_unique), device=sorted_bins.device)
    last_index_incoming_unique = 1 + incoming_unique[-1]

    incoming_to_contigous = torch.zeros(last_index_incoming_unique, dtype=torch.long, device=sorted_bins.device)
    incoming_to_contigous.scatter_reduce_(
        dim=0,
        index=incoming_unique,
        src=incoming_index_range,
        reduce="min",
        include_self=False,
    )

    contigous_distribution_index = torch.gather(incoming_to_contigous, dim=0, index=distribution_index)

    n_valid_distributions = len(incoming_unique)
    packed_info: jt.Int[torch.Tensor, "n_distributions 2"] = nerfacc.pack_info(
        contigous_distribution_index, n_valid_distributions
    )

    edge_cdf: jt.Float[torch.Tensor, "n_bins 2"] = torch.stack(
        [nerfacc.exclusive_sum(pdf, packed_info), nerfacc.inclusive_sum(pdf, packed_info)],
        dim=1,
    )

    is_left: jt.Bool[torch.Tensor, "2 * n_bins"] = torch.tensor(
        [1, 0], device=sorted_bins.device, dtype=torch.bool
    ).repeat(len(sorted_bins))
    double_packed_info = nerfacc.pack_info(
        torch.repeat_interleave(contigous_distribution_index, 2), n_valid_distributions
    )

    intervals = nerfacc.RayIntervals(
        vals=sorted_bins.view(-1),
        packed_info=double_packed_info,
        is_left=is_left,
        is_right=torch.logical_not(is_left),
    )

    if isinstance(n_samples, float):  # add assert  # the whole thing is broken now
        n_points_per_distribution = packed_info[:, 1]
        nerfacc_n_samples = (n_samples * n_points_per_distribution).long()
    elif isinstance(n_samples, int):
        # nerfacc_n_samples = torch.full((n_valid_distributions,), fill_value=n_samples, dtype=torch.long, device=sorted_bins.device)
        nerfacc_n_samples = n_samples
    else:
        raise NotImplementedError

    cuts, _ = nerfacc.importance_sampling(intervals, edge_cdf.view(-1), nerfacc_n_samples)

    subcuts = torch.stack([cuts.vals[:, :-1].flatten(), cuts.vals[:, 1:].flatten()], dim=1)
    return subcuts, torch.repeat_interleave(incoming_unique, n_samples)


def process_in_chunks(*argv: torch.Tensor, process_fn: Callable[..., torch.Tensor], chunk_size: int = 1024 * 32):
    """Process samples (points, rays, etc) in chunks to avoid OOM."""
    num_samples = argv[0].shape[0]
    for a in argv:
        assert a.shape[0] == num_samples, "All input tensors must have the same number of samples."

    results = [process_fn(*[a[i : i + chunk_size] for a in argv]) for i in range(0, num_samples, chunk_size)]

    return torch.concatenate(results, dim=0)


def accumulate_values_from_coo(
    values: jt.Shaped[torch.Tensor, "N *left_dims"],
    coo: jt.Int[torch.Tensor, "N indexed_dims"],
    target: jt.Shaped[torch.Tensor, "*n_dims"],
) -> jt.Shaped[torch.Tensor, "*n_dims"]:
    n_dims = coo.shape[1]
    flat_target_shape = (math.prod(target.shape[:n_dims]), *target.shape[n_dims:])

    flat_target_index = ravel_index(coo, shape=target.shape[:n_dims])
    for dim_index, dim_shape in enumerate(target.shape[n_dims:]):
        running_shape = flat_target_index.shape
        flat_target_index = flat_target_index.unsqueeze_(1 + dim_index)
        running_shape = (*running_shape, dim_shape)
        flat_target_index = flat_target_index.expand(*running_shape)

    return target.view(flat_target_shape).scatter(dim=0, index=flat_target_index, src=values).view(target.shape)


@torch.no_grad()
def tensor_as_image(
    inp: torch.Tensor,
    layout: Literal["CHW", "HWC", "HW"],
    clip: bool = True,
) -> jt.UInt8[np.ndarray, "H W C"]:
    assert len(inp.shape) in (2, 3)
    if layout == "CHW":
        inp = inp.permute(1, 2, 0)

    elif layout == "HW":
        assert len(inp.shape) == 2
        inp = inp.unsqueeze(2).repeat(1, 1, 3)

    inp_scaled = inp / inp.max()
    if clip:
        inp_scaled = torch.clamp(inp_scaled, min=0, max=1)

    return (pt2np(inp_scaled) * 255).astype(np.uint8)


def save_image_tensor(inp: torch.Tensor, path: str, layout: Literal["CHW", "HWC", "HW"]):
    cv2.imwrite(path, cv2.cvtColor(tensor_as_image(inp, layout), cv2.COLOR_RGB2BGR))


def extract_pose_and_rodrigues(world_to_cam):
    # Ensure the matrix is a numpy array
    world_to_cam = np.array(world_to_cam)

    # Extract translation vector
    translation = world_to_cam[:3, 3]

    # Extract rotation matrix
    rotation_matrix = world_to_cam[:3, :3]

    # Convert rotation matrix to Rodriguez rotation vector
    rotation = R.from_matrix(rotation_matrix)
    rodrigues_rotation = rotation.as_rotvec()

    return translation.astype(np.float32), rodrigues_rotation.astype(np.float32)

def ensure_rgb(
    image: Union[jt.UInt8[np.ndarray, "H W C"], jt.UInt8[np.ndarray, "H W"]],
) -> jt.UInt8[np.ndarray, "H W 3"]:
    assert len(image.shape) in (2, 3)

    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    return image

def get_minmax_aabb(
    vertices: jt.Float[torch.Tensor, "N D"],
    margin: Union[float, jt.Float[torch.Tensor, "D"]] = 0.0,
) -> jt.Float[torch.Tensor, "2*D"]:
    v_min, _ = vertices.min(dim=0)
    v_max, _ = vertices.max(dim=0)

    return torch.cat([v_min - margin, v_max + margin], dim=0)

def get_world_ray_pixel_mask(
    vertices_xyz: jt.Float[torch.Tensor, "P 3"],
    K: jt.Float[torch.Tensor, "3 3"],
    Rt: jt.Float[torch.Tensor, "3 4"],
    aabb_margin: float,
    mask_shape: Tuple[int, int],
) -> jt.Bool[torch.Tensor, "H W"]:
    aabb_points = get_minmax_aabb(vertices_xyz, margin=aabb_margin).view(2, 3)
    cartesian_index = torch.cartesian_prod(torch.arange(2), torch.arange(2), torch.arange(2)).to(aabb_points.device)
    box_points = aabb_points.gather(dim=0, index=cartesian_index)
    return camera.get_convex_mask_from_points(K, Rt, box_points, mask_shape)