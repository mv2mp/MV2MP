from typing import Tuple

import jaxtyping as jt
import numpy as np
import torch
import typing_extensions as te
from skimage import morphology

from . import utils

def project_points_to_camera(
    K: jt.Float[torch.Tensor, "3 3"],
    Rt: jt.Float[torch.Tensor, "3 4"],
    points_3d: jt.Float[torch.Tensor, "N 3"],
) -> jt.Float[torch.Tensor, "N 2"]:
    """this function doesn't perform any checks for coordinated to be in image
    for this use `cast_plane_points_to_camera_pixels`

    Returns:
        2d float coordinates
    """
    cam_points = utils.to_homogeneous(points_3d) @ Rt.T

    normalized_cam_points = cam_points / cam_points[:, 2:]
    pixel_points = normalized_cam_points @ K.T

    return pixel_points[:, :2].clone()


def cast_plane_points_to_camera_pixels(
    plane_points: jt.Float[torch.Tensor, "N 2"],
    shape: te.Annotated[Tuple[int, int], "H W"],
) -> Tuple[
    te.Annotated[jt.Int[torch.Tensor, "N 2"], "all points"],
    te.Annotated[jt.Bool[torch.Tensor, "N"], "image hit mask"],
]:
    h, w = shape
    x, y = plane_points[:, 0], plane_points[:, 1]
    mask = (x >= 0) & (x < w) & (y >= 0) & (y < h)

    return plane_points.int(), mask


def project_points_to_camera_np(
    K: jt.Float[np.ndarray, "3 3"],
    Rt: jt.Float[np.ndarray, "3 4"],
    points_3d: jt.Float[np.ndarray, "N 3"],
) -> jt.Float[np.ndarray, "N 2"]:
    cam_points = utils.to_homogeneous_np(points_3d) @ Rt.T

    normalized_cam_points = cam_points / cam_points[:, 2:]
    pixel_points = normalized_cam_points @ K.T

    return pixel_points[:, :2].copy()


def get_point_mask_on_camera_plane_from_points(
    K: jt.Float[torch.Tensor, "3 3"],
    Rt: jt.Float[torch.Tensor, "3 4"],
    points_3d: jt.Float[torch.Tensor, "N 3"],
    mask_shape: te.Annotated[Tuple[int, int], "H W"],
    clamp: bool = False,
) -> jt.Bool[torch.Tensor, "H W"]:
    assert len(mask_shape) == 2
    h, w = mask_shape

    projected_points = project_points_to_camera(K, Rt, points_3d)
    if clamp:
        projected_points.clamp_(
            min=torch.tensor([0, 0], device=K.device),
            max=torch.tensor([w - 1, h - 1], device=K.device),
        )

    all_cam_points, cam_hit_mask = cast_plane_points_to_camera_pixels(
        plane_points=projected_points,
        shape=mask_shape,
    )
    valid_pixels = all_cam_points[cam_hit_mask]
    flat_index = np.ravel_multi_index(utils.pt2np(valid_pixels.flip(1)).T, (h, w))

    mask = np.zeros(w * h, dtype=bool)
    np.put(mask, flat_index, 1)

    return torch.from_numpy(mask.reshape(h, w)).to(points_3d.device)


def get_convex_mask_from_points(
    K: jt.Float[torch.Tensor, "3 3"],
    Rt: jt.Float[torch.Tensor, "3 4"],
    points_3d: jt.Float[torch.Tensor, "N 3"],
    mask_shape: te.Annotated[Tuple[int, int], "H W"],
) -> jt.Bool[torch.Tensor, "H W"]:
    projected_points = get_point_mask_on_camera_plane_from_points(K, Rt, points_3d, mask_shape, clamp=True)
    return torch.from_numpy(morphology.convex_hull_image(utils.pt2np(projected_points))).to(points_3d.device)


@utils.optionally_deterministic
def sample_camera_plane_pixels(
    pixel_mask: jt.Bool[torch.Tensor, "N H W"],
    num_rays_per_camera: int = 64,
    # TODO: sample_pixel_once: bool = True,
) -> jt.Int[torch.Tensor, "P 3"]:
    mask_b, mask_h, mask_w = pixel_mask.shape
    N, Y, X = torch.meshgrid(torch.arange(mask_b), torch.arange(mask_h), torch.arange(mask_w), indexing="ij")

    num_positivie_pixels: jt.Int[torch.Tensor, "N"] = pixel_mask.sum(dim=(1, 2))

    BXY_stacked = torch.stack([N, X, Y], dim=-1).to(pixel_mask.device)
    masked_BXY: jt.Int[torch.Tensor, "M 3"] = BXY_stacked[pixel_mask]

    selected_indices = utils.get_random_coo_indexing_array(
        batch_lengths=num_positivie_pixels,
        num_samples_per_instance=num_rays_per_camera,
    )

    selected_BXY = masked_BXY[selected_indices]

    return selected_BXY
