import os
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import jaxtyping as jt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing_extensions as te
import kaolin

from nerfacc import pack

from lib.model.networks import ImplicitNet, RenderingNet, ConditioningType, KPlanes
from lib.model.density import LaplaceDensity, AbsDensity
from lib.model.ray_sampler import ErrorBoundSampler
from lib.model.deformer import SMPLDeformer
from lib.model.smpl import SMPLServer
from lib.model.sampler import PointInSpace

from lib.utils import utils


def merge_and_sort_points(
    points: List[jt.Float[torch.Tensor, "N"]],
    cut_ray_index: List[jt.Int[torch.Tensor, "N"]],
) -> Tuple[
    te.Annotated[jt.Float[torch.Tensor, "sum_N_i"], "points"],
    te.Annotated[jt.Int[torch.Tensor, "sum_N_i"], "aabb index"],
    te.Annotated[jt.Int[torch.Tensor, "sum_N_i"], "ray index"],
]:
    person_index = [
        i + torch.zeros(aabb_points.shape[0], dtype=torch.int32, device=points[0].device)
        for i, aabb_points in enumerate(points)
    ]
    person_index = torch.cat(person_index)

    ray_index = torch.cat(cut_ray_index)
    points = torch.cat(points)

    sorted_index = points.argsort()
    sorted_index = sorted_index[ray_index[sorted_index].argsort(stable=True)]

    ray_index = ray_index[sorted_index]
    person_index = person_index[sorted_index]
    points = points[sorted_index]

    return points, person_index, ray_index, sorted_index


def get_pad_points_index(
    points: jt.Float[torch.Tensor, "N"], ray_index: jt.Int[torch.Tensor, "N"], n_rays: int, n_steps_per_ray: int
) -> te.Annotated[jt.Int[torch.Tensor, "N_RAYS N_STEPS_PER_RAY"], "pad_points_index"]:
    pad_points_index = torch.full((n_rays, n_steps_per_ray), -1, dtype=torch.int32, device=ray_index.device)
    pack_info = pack.pack_info(ray_index, n_rays).unbind(dim=-1)
    indexes = torch.arange(len(points), dtype=torch.int32, device=points.device)

    cumsum = torch.zeros_like(pack_info[0])
    cumsum[1:] = pack_info[0][1:]

    column_index = torch.repeat_interleave(cumsum, pack_info[1])
    column_index = indexes - column_index

    pad_points_index[ray_index, column_index] = indexes

    return pad_points_index


def get_padded_points(
    points: List[jt.Float[torch.Tensor, "N"]],
    cut_ray_index: List[jt.Int[torch.Tensor, "N"]],
    n_rays: int,
    n_steps_per_ray: int = 128,
) -> Tuple[
    te.Annotated[jt.Float[torch.Tensor, "sum_N_i"], "points"],
    te.Annotated[jt.Int[torch.Tensor, "sum_N_i"], "ray index"],
    te.Annotated[jt.Int[torch.Tensor, "M 2"], "Index of close points"],
    te.Annotated[jt.Int[torch.Tensor, "K 2"], "Index of boundary points"],
]:
    points, person_index, ray_index, sorted_index = merge_and_sort_points(points, cut_ray_index)
    pad_points_index = get_pad_points_index(points, ray_index, n_rays, n_steps_per_ray)

    return points, ray_index, person_index, pad_points_index, sorted_index


import torch
import numpy as np
import torch.nn as nn


class V2A(nn.Module):
    def __init__(
        self,
        opt,
        num_training_frames: int,
        frames: List[int],
        cameras: List[str],
        betas: List[jt.Float[torch.Tensor, "1 10"]],
        genders: List[Literal["male", "female", "neutral"]],
    ):
        print(len(betas))
        super().__init__()

        self.frames = frames
        self.cameras = cameras
        self.wo_merge = opt.wo_merge
        self.model_background = opt.model_background
        # Foreground networks
        self.persons_count = len(genders)
        self.implicit_networks: Sequence[ImplicitNet] = nn.ModuleList()
        self.rendering_networks: Sequence[RenderingNet] = nn.ModuleList()
        for _ in range(self.persons_count):
            self.implicit_networks.append(ImplicitNet(opt.implicit_network))
            self.rendering_networks.append(RenderingNet(opt.rendering_network))

        if self.model_background:
            self.bg_networks: Sequence[KPlanes] = nn.ModuleList()
            for _ in self.cameras:
                k_planes = KPlanes(
                    config=opt.bg_rendering_network,
                    frames_num=len(self.frames),
                )

                self.bg_networks.append(k_planes)

        self.sampler = PointInSpace()

        self.genders = genders
        self.deformers: Sequence[SMPLDeformer] = nn.ModuleList(
            SMPLDeformer(betas=beta, gender=gender) for beta, gender in zip(betas, self.genders)
        )

        # pre-defined bounding sphere
        self.sdf_bounding_sphere = opt.sdf_bounding_sphere_radius

        # threshold for the out-surface points
        self.threshold = 0.05

        self.density = LaplaceDensity(**opt.density)
        self.bg_density = AbsDensity()

        self.ray_sampler = ErrorBoundSampler(self.sdf_bounding_sphere, inverse_sphere_bg=False, **opt.ray_sampler)
        self.smpl_server_list: Sequence[SMPLServer] = nn.ModuleList()
        for i in range(self.persons_count):
            self.smpl_server_list.append(SMPLServer(gender=self.genders[i], betas=betas[i]))

        if opt.smpl_init:
            smpl_model_state = torch.load(f"{os.path.dirname(os.path.abspath(__file__))}/smpl_init.pth")
            for i in range(self.persons_count):
                self.implicit_networks[i].load_state_dict(smpl_model_state["model_state_dict"])

        for person_number in range(self.persons_count):
            self.register_buffer(f"smpl_v_cano_list{person_number}", self.smpl_server_list[person_number].verts_c)
            self.register_buffer(
                f"smpl_f_cano_list{person_number}",
                torch.from_numpy(self.smpl_server_list[person_number].smpl.faces.astype(np.int64)),
            )

            self.set_mesh_vf_cano(
                person_number=person_number,
                v=self.smpl_server_list[person_number].verts_c,
                f=torch.from_numpy(self.smpl_server_list[person_number].smpl.faces.astype(np.int64)),
            )

        self.agg_func = opt.agg_func

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for person_number in range(self.persons_count):
            self.set_mesh_vf_cano(
                person_number,
                state_dict[f"{prefix}mesh_v_cano_list{person_number}"],
                state_dict[f"{prefix}mesh_f_cano_list{person_number}"],
            )

        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def get_smpl_vf_cano(self, person_number: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            getattr(self, f"smpl_v_cano_list{person_number}"),
            getattr(self, f"smpl_f_cano_list{person_number}"),
        )

    def get_mesh_vf_cano(self, person_number: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            getattr(self, f"mesh_v_cano_list{person_number}"),
            getattr(self, f"mesh_f_cano_list{person_number}"),
        )

    def get_mesh_cano_face_vertices_list(self, person_number: int) -> torch.Tensor:
        return getattr(self, f"mesh_face_vertices_list{person_number}")

    def set_mesh_vf_cano(self, person_number: int, v: torch.Tensor, f: torch.Tensor):
        self.register_buffer(f"mesh_v_cano_list{person_number}", v)
        self.register_buffer(f"mesh_f_cano_list{person_number}", f)
        self.register_buffer(f"mesh_face_vertices_list{person_number}", kaolin.ops.mesh.index_vertices_by_faces(v, f))

    def sdf_func_with_smpl_deformer(
        self,
        person_number: int,
        x: jt.Float[torch.Tensor, "N 3"],
        cond: ConditioningType,
        smpl_tfs: jt.Float[torch.Tensor, "J 4 4"],
        smpl_verts,
    ):
        x_c, outlier_mask = self.deformers[person_number].reverse_skinning(
            x.unsqueeze(0), smpl_tfs, smpl_verts=smpl_verts
        )
        outlier_mask = outlier_mask.squeeze(0)
        x_c = x_c.squeeze(0)

        output = self.implicit_networks[person_number](x_c, cond)[0]
        sdf = output[:, 0:1]
        feature = output[:, 1:]
        if not self.training:
            sdf[outlier_mask] = 4.0  # set a large SDF value for outlier points

        return sdf, x_c.squeeze(0), feature

    def check_off_in_surface_points_cano_mesh(self, person_number, x_cano, N_samples, threshold=0.05):
        distance, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
            x_cano.unsqueeze(0).contiguous(), self.get_mesh_cano_face_vertices_list(person_number)
        )

        distance = torch.sqrt(distance)  # kaolin outputs squared distance
        cano_v, cano_f = self.get_mesh_vf_cano(person_number)

        sign = kaolin.ops.mesh.check_sign(cano_v, cano_f, x_cano.unsqueeze(0)).float()
        sign = 1 - 2 * sign
        signed_distance = sign * distance
        batch_size = x_cano.shape[0] // N_samples
        signed_distance = signed_distance.reshape(batch_size, N_samples, 1)

        minimum = torch.min(signed_distance, 1)[0]
        index_off_surface = (minimum > threshold).squeeze(1)
        index_in_surface = (minimum <= 0.0).squeeze(1)
        return index_off_surface, index_in_surface

    def obtain_condition(self, smpl_pose, current_epoch):
        cond = {"smpl": smpl_pose[:, 3:] / np.pi}
        epoch_num_to_refresh = 5
        if self.training:
            # if False:
            if current_epoch < epoch_num_to_refresh or current_epoch % epoch_num_to_refresh == 0:
                cond = {"smpl": smpl_pose[:, 3:] * 0.0}
        return cond

    def front_z_vals_one_person(
        self, input, person_number, ray_dirs, cam_loc, scale, smpl_pose, smpl_shape, smpl_trans
    ):
        smpl_output = self.smpl_server_list[person_number](scale, smpl_trans, smpl_pose, smpl_shape)

        smpl_tfs = smpl_output["smpl_tfs"]
        cond = self.obtain_condition(smpl_pose, input["current_epoch"])

        z_vals, _ = self.ray_sampler.get_z_vals(
            person_number, ray_dirs, cam_loc, self, cond, smpl_tfs, eval_mode=True, smpl_verts=smpl_output["smpl_verts"]
        )
        return z_vals, smpl_output

    def forward_color_occupancy_for_one_person(
        self,
        person_number: int,
        ray_dirs: jt.Float[torch.Tensor, "num_pixels 3"],
        points: jt.Float[torch.Tensor, "num_pixels n_samples 3"],
        smpl_shape: jt.Float[torch.Tensor, "1 69"],
        smpl_tfs: jt.Float[torch.Tensor, "1 24 4 4"],
        smpl_vertices: jt.Float[torch.Tensor, "1 6890 3"],
    ) -> Tuple[
        jt.Float[torch.Tensor, "num_pixels*n_samples 1"],
        jt.Float[torch.Tensor, "num_pixels n_samples 3"],
        jt.Float[torch.Tensor, "num_pixels n_samples 3"],
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        cond = {"smpl": smpl_shape}
        batch_size = 1
        num_pixels, _ = ray_dirs.shape
        ray_dirs = ray_dirs.reshape(-1, 3)

        N_samples = points.shape[1]

        points_flat = points.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1, N_samples, 1)
        sdf_output, canonical_points, feature_vectors = self.sdf_func_with_smpl_deformer(
            person_number,
            points_flat,
            cond,
            smpl_tfs,
            smpl_vertices,
        )

        sdf_output = sdf_output.unsqueeze(1)
        index_off_surface = None
        index_in_surface = None
        if self.training:
            index_off_surface, index_in_surface = self.check_off_in_surface_points_cano_mesh(
                person_number, canonical_points, N_samples, threshold=self.threshold
            )
            canonical_points = canonical_points.reshape(num_pixels, N_samples, 3)

            canonical_points = canonical_points.reshape(-1, 3)

            # sample canonical SMPL surface pnts for the eikonal loss
            smpl_verts_c = self.smpl_server_list[person_number].verts_c.repeat(batch_size, 1, 1)

            indices = torch.randperm(smpl_verts_c.shape[1])[:num_pixels].cuda()
            verts_c = torch.index_select(smpl_verts_c, 1, indices)
            sample = self.sampler.get_points(verts_c, global_ratio=0.0)

            sample.requires_grad_()
            local_pred = self.implicit_networks[person_number](sample, cond)[..., 0:1]
            grad_theta = gradient(sample, local_pred)
        else:
            canonical_points = canonical_points.reshape(num_pixels, N_samples, 3).reshape(-1, 3)
            grad_theta = None

        sdf_output = sdf_output.reshape(num_pixels, N_samples, 1).reshape(-1, 1)
        view = -dirs.reshape(-1, 3)

        if canonical_points.shape[0] > 0:
            fg_rgb_flat, others = self.get_rgb_value(
                person_number,
                canonical_points,
                view,
                cond,
                smpl_tfs,
                feature_vectors=feature_vectors,
                is_training=self.training,
            )
            normal_values = others["normals"]

        fg_rgb = fg_rgb_flat.reshape(-1, N_samples, 3)
        normal_values = normal_values.reshape(-1, N_samples, 3)
        return sdf_output, fg_rgb, normal_values, grad_theta, index_in_surface, index_off_surface

    def get_inputs(self, input, person_number):
        scale = input["smpl_params"][:, person_number, 0]
        smpl_pose = input["smpl_pose"][:, person_number, :]
        smpl_shape = input["smpl_shape"][:, person_number, :]
        smpl_trans = input["smpl_trans"][:, person_number, :]
        return scale, smpl_pose, smpl_shape, smpl_trans

    def forward_standard_merge(self, input):
        intrinsics = input["intrinsics"]
        pose = input["pose"]
        uv = input["uv"]

        ray_dirs, cam_loc = utils.get_camera_params(uv, pose, intrinsics)
        _, num_pixels, _ = ray_dirs.shape
        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        z_vals_list = []
        smpl_output_list = []

        cond_list = []
        for person_number in range(self.persons_count):
            scale, smpl_pose, smpl_shape, smpl_trans = self.get_inputs(input, person_number)
            cond_list.append(self.obtain_condition(smpl_pose, input["current_epoch"]))
            person_z_vals, smpl_output = self.front_z_vals_one_person(
                input, person_number, ray_dirs, cam_loc, scale, smpl_pose, smpl_shape, smpl_trans
            )
            z_vals_list.append(person_z_vals)
            smpl_output_list.append(smpl_output)

        sdf_output_list = []
        fg_rgb_list = []
        normal_values_list = []
        grad_theta_list = []
        index_in_surface_list = []
        index_off_surface_list = []
        density_list = []

        for person_number in range(self.persons_count):
            person_points = cam_loc.unsqueeze(1) + z_vals_list[person_number].unsqueeze(2) * ray_dirs.unsqueeze(1)
            cond = cond_list[person_number]
            smpl_output = smpl_output_list[person_number]
            (
                sdf_output,
                fg_rgb,
                normal_values,
                grad_theta,
                index_in_surface,
                index_off_surface,
            ) = self.forward_color_occupancy_for_one_person(
                person_number,
                ray_dirs,
                person_points,
                cond["smpl"],
                smpl_output["smpl_tfs"],
                smpl_output["smpl_verts"],
            )

            sdf_output_list.append(sdf_output)
            fg_rgb_list.append(fg_rgb)
            normal_values_list.append(normal_values)
            grad_theta_list.append(grad_theta)
            index_in_surface_list.append(index_in_surface)
            index_off_surface_list.append(index_off_surface)

            density_flat = self.density(sdf_output)  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
            density = density_flat.reshape(
                -1, z_vals_list[person_number].shape[1]
            )  # (batch_size * num_pixels) x N_samples
            density_list.append(density)

        # rendering    #####   #####   #####   #####   #####   #####
        merged_z_vals, sorted_indices = torch.sort(
            torch.cat(z_vals_list, dim=-1), dim=-1
        )  # sorted indices are the indices that i can query z_vals_list and get sorted list.

        def sort3_with2(input, sorted_indices):
            R, P, C = input.shape
            r_indices = torch.arange(R)[:, None, None].expand(-1, P, C)
            color_indices = torch.arange(C)[None, None, :].expand(R, P, -1)

            # Use the sort_index for the S dimension and expand it for 3 channels
            sorted_s_indices = sorted_indices[:, :, None].expand(-1, -1, C)

            # Apply advanced indexing to sort the colors tensor
            return input[r_indices, sorted_s_indices, color_indices]

        def sort2_with2(input, sorted_indices):
            R, P = input.shape
            r_indices = torch.arange(R)[:, None].expand(-1, P)
            return input[r_indices, sorted_indices]

        cat_colors = torch.cat(fg_rgb_list, dim=-2)
        sorted_colors = sort3_with2(cat_colors, sorted_indices)

        cat_densities = torch.cat(density_list, dim=-1)
        sorted_cat_densities = sort2_with2(cat_densities, sorted_indices)

        # included also the dist from the sphere intersection
        dists = merged_z_vals[:, 1:] - merged_z_vals[:, :-1]

        free_energy = dists * sorted_cat_densities[:, :-1]

        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy], dim=-1)
        alpha = 1 - torch.exp(-free_energy)

        transmittance = torch.exp(
            -torch.cumsum(shifted_free_energy, dim=-1)
        )  # probability of everything is empty up to now
        fg_transmittance = transmittance[:, :-1]

        weights = alpha * fg_transmittance
        rgb_values = torch.sum(weights.unsqueeze(-1) * sorted_colors[:, :-1, :], 1)
        acc_map = torch.sum(weights, -1)

        if self.model_background:
            frame_idx = input["frame_num"]
            camera_idx = self.cameras.index(input["camera_name"][0])
            t = torch.broadcast_to(torch.tensor([frame_idx]), (uv.shape[1], -1)).to(uv.device)
            u_norm = uv[0, :, 0:1] / input["img_size"][1]
            v_norm = uv[0, :, 1:2] / input["img_size"][0]
            xyt = torch.cat([u_norm, v_norm, t], dim=-1)
            bg_weights = torch.clamp(1 - acc_map, min=0, max=1)
            bg_colors = self.bg_networks[camera_idx](xyt)

            rgb_values += bg_weights[..., None] * bg_colors

        # end of rendering    #####   #####   #####   #####   #####   #####

        ########now collect sdfs for sdf loss ######
        # dist_common = merged_z_vals[:, 1:] - merged_z_vals[:, :-1]
        if self.persons_count > 2:
            sdf_for_loss = []
        else:
            sdf1 = sdf_output_list[0].reshape(density_list[0].shape)  # Rays x Points
            sdf2 = sdf_output_list[1].reshape(density_list[1].shape)  # Rays x Points
            # collect first

            sdf2_zeros = torch.zeros_like(sdf2)
            sdf1_full = torch.cat([sdf1, sdf2_zeros], dim=-1)
            sdf1_sorted = sort2_with2(sdf1_full, sorted_indices)

            sdf1_mask = torch.cat([torch.ones_like(sdf1), torch.zeros_like(sdf2)], -1)
            sdf1_mask_sorted = sort2_with2(sdf1_mask, sorted_indices) == 1

            sdf1_sorted_left = torch.zeros_like(sdf1_sorted)
            sdf1_sorted_left[:, 1:] = sdf1_sorted[:, :-1]
            sdf1_sorted_left[:, 1:] += dists

            sdf1_sorted_right = torch.zeros_like(sdf1_sorted)
            sdf1_sorted_right[:, :-1] = sdf1_sorted[:, 1:]
            sdf1_sorted_right[:, :-1] += dists

            sdf1_left_right = torch.min(sdf1_sorted_left, sdf1_sorted_right)
            sdf1_sorted[~sdf1_mask_sorted] = sdf1_left_right[~sdf1_mask_sorted]

            # collect second

            sdf1_zeros = torch.zeros_like(sdf1)
            sdf2_full = torch.cat([sdf1_zeros, sdf2], dim=-1)
            sdf2_sorted = sort2_with2(sdf2_full, sorted_indices)

            sdf2_mask = torch.cat([torch.zeros_like(sdf1), torch.ones_like(sdf2)], -1)
            sdf2_mask_sorted = sort2_with2(sdf2_mask, sorted_indices) == 1

            sdf2_sorted_left = torch.zeros_like(sdf2_sorted)
            sdf2_sorted_left[:, 1:] = sdf2_sorted[:, :-1]
            sdf2_sorted_left[:, 1:] += dists

            sdf2_sorted_right = torch.zeros_like(sdf2_sorted)
            sdf2_sorted_right[:, :-1] = sdf2_sorted[:, 1:]
            sdf2_sorted_right[:, :-1] += dists

            sdf2_left_right = torch.min(sdf2_sorted_left, sdf2_sorted_right)
            sdf2_sorted[~sdf2_mask_sorted] = sdf2_left_right[~sdf2_mask_sorted]
            sdf_for_loss = [sdf1_sorted, sdf2_sorted]

        ########end now collect sdfs for sdf loss ######

        if self.training:
            output = {
                # "points": ,
                "rgb_values": rgb_values,
                # 'normal_values_list': normal_values_list,
                "index_outside": input["index_outside"],
                "index_off_surface_list": index_off_surface_list,
                "index_in_surface_list": index_in_surface_list,
                "acc_map": acc_map,
                "sdf_output_list": sdf_output_list,
                "grad_theta_list": grad_theta_list,
                "epoch": input["current_epoch"],
                "sdf_for_loss": sdf_for_loss
                # 'canonical_points': canonical_points,
                # 'smpl_output': smpl_output,
            }
        else:
            output = {
                "acc_map": acc_map,
                "rgb_values": rgb_values,
                "fg_rgb_values": rgb_values,
                # 'normal_values_list': normal_values_list,
                "sdf_output_list": sdf_output_list,
                # 'canonical_points':canonical_points,
                # 'smpl_output' : smpl_output,
            }
        return output

    def forward(self, input):
        if self.wo_merge:
            return self.forward_standard_merge(input)

        intrinsics = input["intrinsics"]
        pose = input["pose"]
        uv = input["uv"]

        ray_dirs, cam_loc = utils.get_camera_params(uv, pose, intrinsics)
        _, num_pixels, _ = ray_dirs.shape
        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        z_vals_list = []
        smpl_output_list = []

        cond_list = []
        for person_number in range(self.persons_count):
            scale, smpl_pose, smpl_shape, smpl_trans = self.get_inputs(input, person_number)
            cond_list.append(self.obtain_condition(smpl_pose, input["current_epoch"]))
            person_z_vals, smpl_output = self.front_z_vals_one_person(
                input, person_number, ray_dirs, cam_loc, scale, smpl_pose, smpl_shape, smpl_trans
            )
            z_vals_list.append(person_z_vals)
            smpl_output_list.append(smpl_output)

        merged_z_vals, _ = torch.sort(torch.cat(z_vals_list, dim=-1), dim=-1)
        merged_z_max = merged_z_vals[:, -1]
        merged_z_vals = merged_z_vals[:, :-1]

        merged_points = cam_loc.unsqueeze(1) + merged_z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)

        sdf_output_list = []
        fg_rgb_list = []
        normal_values_list = []
        grad_theta_list = []
        index_in_surface_list = []
        index_off_surface_list = []

        # included also the dist from the sphere intersection
        dists = merged_z_vals[:, 1:] - merged_z_vals[:, :-1]
        dists = torch.cat([dists, merged_z_max.unsqueeze(-1) - merged_z_vals[:, -1:]], -1)

        shifted_free_energy_list = []
        alpha_list = []

        for person_number in range(self.persons_count):
            cond = cond_list[person_number]
            smpl_output = smpl_output_list[person_number]
            (
                sdf_output,
                fg_rgb,
                normal_values,
                grad_theta,
                index_in_surface,
                index_off_surface,
            ) = self.forward_color_occupancy_for_one_person(
                person_number,
                ray_dirs,
                merged_points,
                cond["smpl"],
                smpl_output["smpl_tfs"],
                smpl_output["smpl_verts"],
            )

            sdf_output_list.append(sdf_output)
            fg_rgb_list.append(fg_rgb)
            normal_values_list.append(normal_values)
            grad_theta_list.append(grad_theta)
            index_in_surface_list.append(index_in_surface)
            index_off_surface_list.append(index_off_surface)

            density_flat = self.density(sdf_output)  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
            density = density_flat.reshape(-1, merged_z_vals.shape[1])  # (batch_size * num_pixels) x N_samples
            free_energy = dists * density
            shifted_free_energy = torch.cat(
                [torch.zeros(dists.shape[0], 1).cuda(), free_energy], dim=-1
            )  # add 0 for transperancy 1 at t_0
            alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
            shifted_free_energy_list.append(shifted_free_energy)
            alpha_list.append(alpha)

        aggregated_shifted_free_energy = self.aggregate_free_energy(shifted_free_energy_list)
        transmittance = torch.exp(
            -torch.cumsum(aggregated_shifted_free_energy, dim=-1)
        )  # probability of everything is empty up to now
        fg_transmittance = transmittance[:, :-1]

        weight_list = [alpha * fg_transmittance for alpha in alpha_list]

        acc_map = torch.sum(torch.sum(torch.stack(weight_list, dim=0), dim=0), dim=-1)

        rgb_values = torch.zeros([acc_map.shape[0], 3], dtype=torch.float32).to(acc_map.device)
        if self.model_background:
            frame_idx = input["frame_num"]
            camera_idx = self.cameras.index(input["camera_name"][0])
            t = torch.broadcast_to(torch.tensor([frame_idx]), (uv.shape[1], -1)).to(uv.device)
            u_norm = uv[0, :, 0:1] / input["img_size"][1]
            v_norm = uv[0, :, 1:2] / input["img_size"][0]
            xyt = torch.cat([u_norm, v_norm, t], dim=-1)
            bg_weights = torch.clamp(1 - acc_map, min=0, max=1)
            bg_colors = self.bg_networks[camera_idx](xyt)

            rgb_values += bg_weights[..., None] * bg_colors

        for weights, fg_rgb in zip(weight_list, fg_rgb_list):
            person_rgb = torch.sum(weights.unsqueeze(-1) * fg_rgb, dim=1)
            rgb_values += person_rgb

        if self.training:
            output = {
                # "points": merged_points,
                "rgb_values": rgb_values,
                # 'normal_values_list': normal_values_list,
                "index_outside": input["index_outside"],
                "index_off_surface_list": index_off_surface_list,
                "index_in_surface_list": index_in_surface_list,
                "acc_map": acc_map,
                "sdf_output_list": sdf_output_list,
                "grad_theta_list": grad_theta_list,
                "epoch": input["current_epoch"],
                "sdf_for_loss": sdf_output_list
                # 'canonical_points': canonical_points,
                # 'smpl_output': smpl_output,
            }
        else:
            output = {
                "acc_map": acc_map,
                "rgb_values": rgb_values,
                "fg_rgb_values": rgb_values,
                # 'normal_values_list': normal_values_list,
                "sdf_output_list": sdf_output_list,
                # 'canonical_points':canonical_points,
                # 'smpl_output' : smpl_output,
            }
        return output

    def aggregate_free_energy(self, free_energy_list):
        stacked_tensors = torch.stack(free_energy_list, dim=0)

        if self.agg_func == "mean":
            return torch.mean(stacked_tensors, dim=0)
        elif self.agg_func == "max":
            return torch.max(stacked_tensors, dim=0)[0]
        elif self.agg_func == "sum":
            return torch.sum(stacked_tensors, dim=0)

    def get_rgb_value(
        self,
        person_number: int,
        canonical_points: jt.Float[torch.Tensor, "num_pixels*n_samples 3"],
        view_dirs: jt.Float[torch.Tensor, "num_pixels*n_samples 3"],
        cond: ConditioningType,
        tfs: jt.Float[torch.Tensor, "1 J 4 4"],
        feature_vectors,
        is_training: bool = True,
    ) -> Tuple[
        jt.Float[torch.Tensor, "num_pixels*n_samples 3"],
        Dict[Literal["normal"], jt.Float[torch.Tensor, "num_pixels*n_samples 3"]],
    ]:
        others = {}

        _, gradients, feature_vectors = self.forward_gradient(
            person_number, canonical_points, cond, tfs, create_graph=is_training, retain_graph=is_training
        )
        # ensure the gradient is normalized
        normals = nn.functional.normalize(gradients, dim=-1, eps=1e-6)
        fg_rendering_output = self.rendering_networks[person_number](
            canonical_points, normals, view_dirs, cond["smpl"], feature_vectors
        )

        rgb_vals = fg_rendering_output[:, :3]
        others["normals"] = normals
        return rgb_vals, others

    def forward_gradient(
        self,
        person_number: int,
        canonical_points: jt.Float[torch.Tensor, "N 3"],
        cond: ConditioningType,
        tfs: jt.Float[torch.Tensor, "1 J 4 4"],
        create_graph: bool = True,
        retain_graph: bool = True,
    ):
        if canonical_points.shape[0] == 0:
            return canonical_points.detach()
        canonical_points.requires_grad_(True)
        tfs.requires_grad_(True)

        was_grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        pnts_d, _ = self.deformers[person_number].forward_skinning(
            xc=canonical_points.unsqueeze(0),
            smpl_tfs=tfs,
        )
        pnts_d = pnts_d.squeeze(0)

        num_dim = pnts_d.shape[-1]
        grads = []
        for i in range(num_dim):
            d_out = torch.zeros_like(pnts_d, requires_grad=False, device=pnts_d.device)
            d_out[:, i] = 1
            grad = torch.autograd.grad(
                outputs=pnts_d,
                inputs=canonical_points,
                grad_outputs=d_out,
                create_graph=create_graph,
                retain_graph=True if i < num_dim - 1 else retain_graph,
                only_inputs=True,
            )[0]
            grads.append(grad)
        grads = torch.stack(grads, dim=-2)
        grads_inv = grads.inverse()

        output = self.implicit_networks[person_number].forward(canonical_points, cond)[0]
        sdf = output[:, :1]

        feature = output[:, 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=canonical_points,
            grad_outputs=d_output,
            create_graph=create_graph,
            retain_graph=retain_graph,
            only_inputs=True,
        )[0]

        torch.set_grad_enabled(was_grad_enabled)

        return (
            grads.reshape(grads.shape[0], -1),
            torch.nn.functional.normalize(torch.einsum("bi,bij->bj", gradients, grads_inv), dim=1),
            feature,
        )


def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = torch.autograd.grad(
        outputs=outputs, inputs=inputs, grad_outputs=d_points, create_graph=True, retain_graph=True, only_inputs=True
    )[0][:, :, -3:]
    return points_grad
