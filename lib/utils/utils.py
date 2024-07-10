from typing import Tuple

import jaxtyping as jt
import numpy as np
import cv2
import torch
from torch.nn import functional as F


def get_from_dict_recusrive(d, keys, default=None):
    if isinstance(d, dict):
        if keys[0] in d:
            return get_from_dict_recusrive(d[keys[0]], keys[1:], default)
        else:
            return default
    else:
        return d


def split_input(model_input, total_pixels, n_pixels=10000):
    """
    Split the input to fit Cuda memory for large resolution.
    Can decrease the value of n_pixels in case of cuda out of memory error.
    """

    split = []

    whole_index = torch.arange(total_pixels).cuda()
    for i, indx in enumerate(torch.split(whole_index, n_pixels, dim=0)):
        data = model_input.copy()
        data["uv"] = torch.index_select(model_input["uv"], 1, indx)
        split.append(data)
    return split


def merge_output(res, total_pixels, batch_size):
    """Merge the split output."""

    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, 1) for r in res], 1).reshape(
                batch_size * total_pixels
            )
        else:
            model_outputs[entry] = torch.cat(
                [r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res], 1
            ).reshape(batch_size * total_pixels, -1)
    return model_outputs


def get_psnr(img1, img2, normalize_rgb=False):
    if normalize_rgb:  # [-1,1] --> [0,1]
        img1 = (img1 + 1.0) / 2.0
        img2 = (img2 + 1.0) / 2.0

    mse = torch.mean((img1 - img2) ** 2)
    psnr = -10.0 * torch.log(mse) / torch.log(torch.Tensor([10.0]).cuda())

    return psnr


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def get_camera_params(
    uv: jt.Float[torch.Tensor, "B N 2"],
    pose: jt.Float[torch.Tensor, "B 4 4"],
    intrinsics: jt.Float[torch.Tensor, "B 3 3"],
) -> Tuple[jt.Float[torch.Tensor, "B N 3"], jt.Float[torch.Tensor, "B 3"]]:
    if pose.shape[1] == 7:  # In case of quaternion vector representation
        cam_loc = pose[:, 4:]
        R = quat_to_rot(pose[:, :4])
        p = torch.eye(4).repeat(pose.shape[0], 1, 1).cuda().float()
        p[:, :3, :3] = R
        p[:, :3, 3] = cam_loc
    else:  # In case of pose matrix representation
        cam_loc = pose[:, :3, 3]
        p = pose

    batch_size, num_samples, _ = uv.shape

    depth = torch.ones((batch_size, num_samples)).cuda()
    x_cam = uv[:, :, 0].view(batch_size, -1)
    y_cam = uv[:, :, 1].view(batch_size, -1)
    z_cam = depth.view(batch_size, -1)

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)

    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

    world_coords = torch.bmm(p, pixel_points_cam).permute(0, 2, 1)[:, :, :3]
    ray_dirs = world_coords - cam_loc[:, None, :]
    ray_dirs = F.normalize(ray_dirs, dim=2)

    return ray_dirs, cam_loc


def lift(x, y, z, intrinsics):
    # parse intrinsics
    intrinsics = intrinsics.cuda()
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    x_lift = (
        (
            x
            - cx.unsqueeze(-1)
            + cy.unsqueeze(-1) * sk.unsqueeze(-1) / fy.unsqueeze(-1)
            - sk.unsqueeze(-1) * y / fy.unsqueeze(-1)
        )
        / fx.unsqueeze(-1)
        * z
    )
    y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

    # homogeneous
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z).cuda()), dim=-1)


def quat_to_rot(q):
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3, 3)).cuda()
    qr = q[:, 0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0] = 1 - 2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj * qi - qk * qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1 - 2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2 * (qj * qk - qi * qr)
    R[:, 2, 0] = 2 * (qk * qi - qj * qr)
    R[:, 2, 1] = 2 * (qj * qk + qi * qr)
    R[:, 2, 2] = 1 - 2 * (qi**2 + qj**2)
    return R


def rot_to_quat(R):
    batch_size, _, _ = R.shape
    q = torch.ones((batch_size, 4)).cuda()

    R00 = R[:, 0, 0]
    R01 = R[:, 0, 1]
    R02 = R[:, 0, 2]
    R10 = R[:, 1, 0]
    R11 = R[:, 1, 1]
    R12 = R[:, 1, 2]
    R20 = R[:, 2, 0]
    R21 = R[:, 2, 1]
    R22 = R[:, 2, 2]

    q[:, 0] = torch.sqrt(1.0 + R00 + R11 + R22) / 2
    q[:, 1] = (R21 - R12) / (4 * q[:, 0])
    q[:, 2] = (R02 - R20) / (4 * q[:, 0])
    q[:, 3] = (R10 - R01) / (4 * q[:, 0])
    return q


def get_sphere_intersections(cam_loc, ray_directions, r=1.0):
    # Input: n_rays x 3 ; n_rays x 3
    # Output: n_rays x 1, n_rays x 1 (close and far)

    ray_cam_dot = torch.bmm(ray_directions.view(-1, 1, 3), cam_loc.view(-1, 3, 1)).squeeze(-1)
    under_sqrt = ray_cam_dot**2 - (cam_loc.norm(2, 1, keepdim=True) ** 2 - r**2)

    # sanity check
    if (under_sqrt <= 0).sum() > 0:
        print("BOUNDING SPHERE PROBLEM!")
        exit()

    sphere_intersections = torch.sqrt(under_sqrt) * torch.Tensor([-1, 1]).cuda().float() - ray_cam_dot
    sphere_intersections = sphere_intersections.clamp_min(0.0)

    return sphere_intersections


def bilinear_interpolation(xs, ys, dist_map):
    x1 = np.floor(xs).astype(np.int32)
    y1 = np.floor(ys).astype(np.int32)
    x2 = x1 + 1
    y2 = y1 + 1

    dx = np.expand_dims(np.stack([x2 - xs, xs - x1], axis=1), axis=1)
    dy = np.expand_dims(np.stack([y2 - ys, ys - y1], axis=1), axis=2)
    Q = np.stack([dist_map[x1, y1], dist_map[x1, y2], dist_map[x2, y1], dist_map[x2, y2]], axis=1).reshape(-1, 2, 2)
    return np.squeeze(dx @ Q @ dy)  # ((x2 - x1) * (y2 - y1)) = 1


def get_index_outside_of_bbox(samples_uniform, bbox_min, bbox_max):
    samples_uniform_row = samples_uniform[:, 0]
    samples_uniform_col = samples_uniform[:, 1]
    index_outside = np.where(
        (samples_uniform_row < bbox_min[0])
        | (samples_uniform_row > bbox_max[0])
        | (samples_uniform_col < bbox_min[1])
        | (samples_uniform_col > bbox_max[1])
    )[0]
    return index_outside


def weighted_sampling(data, img_size, num_sample, no_rays_mask=None):
    """
    More sampling within the bounding box
    """

    # calculate bounding box
    foreground_mask = data["foreground_mask"]

    where = np.asarray(np.where(foreground_mask))  # (yx, N)

    where_indexer = np.arange(where.shape[1])

    pixels_to_take = np.random.choice(where_indexer, num_sample, replace=True)
    d2_offsets = np.random.rand(num_sample, 2)
    taken_pixels = np.clip(where[:, pixels_to_take].T - d2_offsets, a_min=0, a_max=np.array(img_size) - 1)

    index_outside = np.array([], dtype=np.int64)
    indices = taken_pixels

    # indices i, j of samples, but i, j are not integer, they are real.
    # try to do a bilinear interpolation of mask of i j s.
    # mask of out of seegmentation is where value is > 0;

    foreground_mask = np.clip(foreground_mask, 0, 1)

    if "background_mask" in data:
        bg_mask = data["background_mask"]
    else:
        bg_mask = 1 - foreground_mask

    samples_mask = bilinear_interpolation(indices[:, 0], indices[:, 1], foreground_mask)
    sample_fg_mask = samples_mask > 0.5

    samples_bg_mask = bilinear_interpolation(indices[:, 0], indices[:, 1], bg_mask)
    sample_bg_mask = samples_bg_mask > 0.5

    if no_rays_mask is not None:
        new_val = bilinear_interpolation(indices[:, 0], indices[:, 1], no_rays_mask)
        good_indices_mask = (new_val - 0) < 0.0001
        sample_range = np.arange(num_sample)
        good_range = sample_range[good_indices_mask]
        sample_count = num_sample - len(good_range)
        good_samples = np.random.choice(good_range, sample_count, replace=True)
        bad_indices_mask = good_indices_mask == False
        indices[bad_indices_mask] = indices[good_samples]

    output = {}
    for key, val in data.items():
        if len(val.shape) == 3:
            new_val = np.stack(
                [bilinear_interpolation(indices[:, 0], indices[:, 1], val[:, :, i]) for i in range(val.shape[2])],
                axis=-1,
            )
        else:
            new_val = bilinear_interpolation(indices[:, 0], indices[:, 1], val)
        new_val = new_val.reshape(-1, *val.shape[2:])
        output[key] = new_val

    return output, index_outside, sample_fg_mask, sample_bg_mask  # uv is xy


def box_mask_from_xyxy(xyxy, img_size_hw):
    mask = np.zeros(img_size_hw, dtype=bool)
    mask[xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]] = True
    return mask
