from typing import Dict, Literal, Optional

import jaxtyping as jt
import torch.nn as nn
import torch
import numpy as np
from lib.model.embedders import get_embedder
import torch.nn.functional as F

ConditioningType = Dict[Literal["smpl", "frame"], torch.Tensor]


class ImplicitNet(nn.Module):
    def __init__(self, opt):
        super().__init__()

        dims = [opt.d_in] + list(opt.dims) + [opt.d_out + opt.feature_vector_size]
        self.num_layers = len(dims)
        self.skip_in = opt.skip_in
        self.embed_fn = None
        self.opt = opt

        if opt.multires > 0:
            embed_fn, input_ch = get_embedder(opt.multires, input_dims=opt.d_in, mode=opt.embedder_mode)
            self.embed_fn = embed_fn
            dims[0] = input_ch
        self.cond = opt.cond
        if self.cond == "smpl":
            self.cond_layer = [0]
            self.cond_dim = 69
        elif self.cond == "frame":
            self.cond_layer = [0]
            self.cond_dim = opt.dim_frame_encoding
        else:
            raise NotImplementedError

        self.dim_pose_embed = 0
        if self.dim_pose_embed > 0:
            self.lin_p0 = nn.Linear(self.cond_dim, self.dim_pose_embed)
            self.cond_dim = self.dim_pose_embed
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            if self.cond != "none" and l in self.cond_layer:
                lin = nn.Linear(dims[l] + self.cond_dim, out_dim)
            else:
                lin = nn.Linear(dims[l], out_dim)
            if opt.init == "geometry":
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -opt.bias)
                elif opt.multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif opt.multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            if opt.init == "zero":
                init_val = 1e-5
                if l == self.num_layers - 2:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.uniform_(lin.weight, -init_val, init_val)
            if opt.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.softplus = nn.Softplus(beta=100)

    def forward(
        self,
        input: torch.Tensor,
        cond: Dict[Literal["smpl", "frame"], torch.Tensor],
        current_epoch: Optional[int] = None,
    ):
        if input.ndim == 2:
            input = input.unsqueeze(0)

        num_batch, num_point, num_dim = input.shape

        if num_batch * num_point == 0:
            return input

        input = input.reshape(num_batch * num_point, num_dim)

        if self.cond != "none":
            num_batch, num_cond = cond[self.cond].shape

            input_cond = cond[self.cond].unsqueeze(1).expand(num_batch, num_point, num_cond)

            input_cond = input_cond.reshape(num_batch * num_point, num_cond)

            if self.dim_pose_embed:
                input_cond = self.lin_p0(input_cond)

        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if self.cond != "none" and l in self.cond_layer:
                x = torch.cat([x, input_cond], dim=-1)
            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.softplus(x)

        x = x.reshape(num_batch, num_point, -1)

        return x

    def gradient(self, x, cond):
        x.requires_grad_(True)
        y = self.forward(x, cond)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y, inputs=x, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        return gradients.unsqueeze(1)


class RenderingNet(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.mode = opt.mode
        dims = [opt.d_in + opt.feature_vector_size] + list(opt.dims) + [opt.d_out]

        self.embedview_fn = None
        if opt.multires_view > 0:
            embedview_fn, input_ch = get_embedder(opt.multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += input_ch - 3
        if self.mode == "nerf_frame_encoding":
            dims[0] += opt.dim_frame_encoding
        if self.mode == "pose":
            self.dim_cond_embed = 8
            self.cond_dim = 69  # dimension of the body pose, global orientation excluded.
            # lower the condition dimension
            self.lin_pose = torch.nn.Linear(self.cond_dim, self.dim_cond_embed)
        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if opt.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, points, normals, view_dirs, body_pose, feature_vectors, frame_latent_code=None):
        if self.embedview_fn is not None:
            if self.mode == "nerf_frame_encoding":
                view_dirs = self.embedview_fn(view_dirs)

        if self.mode == "nerf_frame_encoding":
            frame_latent_code = frame_latent_code.expand(view_dirs.shape[0], -1)
            rendering_input = torch.cat([view_dirs, frame_latent_code, feature_vectors], dim=-1)
        elif self.mode == "pose":
            num_points = points.shape[0]
            body_pose = body_pose.unsqueeze(1).expand(-1, num_points, -1).reshape(num_points, -1)
            body_pose = self.lin_pose(body_pose)
            rendering_input = torch.cat([points, normals, body_pose, feature_vectors], dim=-1)
        else:
            raise NotImplementedError

        x = rendering_input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)
        x = self.sigmoid(x)
        return x


class KPlanes(nn.Module):
    FEATURE_MLP_SIZE = 64

    def __init__(self, config, frames_num):
        """
        The parameter scale represents the maximum absolute value among all coordinates and is used for scaling the data
        """
        super(KPlanes, self).__init__()

        hidden_dim1 = config.hidden_dim1
        hidden_dim2 = config.hidden_dim2
        height_dim = int(config.img_h * config.space_scale)
        width_dim = int(config.img_w * config.space_scale)
        time_dim = int(frames_num * config.time_scale)
        f_size = config.f_size

        self.yx_plane = nn.Parameter(torch.rand((f_size, height_dim, width_dim)))
        self.xt_plane = nn.Parameter(torch.rand((f_size, width_dim, time_dim)))
        self.yt_plane = nn.Parameter(torch.rand((f_size, height_dim, time_dim)))

        self.feature_mlp = nn.Sequential(
            nn.Linear(f_size, hidden_dim1), nn.ReLU(), nn.Linear(hidden_dim1, KPlanes.FEATURE_MLP_SIZE), nn.ReLU()
        )

        uv_pos_enc_size = config.uv_pos_enc_level * 2 * 2 + 2
        t_pos_enc_size = config.t_pos_enc_level * 2 + 1

        self.block2 = nn.Sequential(
            nn.Linear(KPlanes.FEATURE_MLP_SIZE + uv_pos_enc_size + t_pos_enc_size, hidden_dim2 * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim2 * 2, hidden_dim2 * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim2 * 2, hidden_dim2 * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim2 * 2, 3),
            nn.Sigmoid(),
        )

        self.uv_pos_enc_level = config.uv_pos_enc_level
        self.t_pos_enc_level = config.t_pos_enc_level
        self.height_dim = height_dim
        self.width_dim = width_dim
        self.time_dim = time_dim
        self.frames_num = frames_num

    def positional_encoding(self, x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2**j * x))
            out.append(torch.cos(2**j * x))
        return torch.cat(out, dim=-1)

    def plane_sample(self, plane, uv):
        # uv - h,w
        grid = uv[None, :, None, :] * 2 - 1
        nchwin = plane[None]
        result = F.grid_sample(nchwin, grid, padding_mode="border")
        res_colors = result[0, :, :, 0].permute(1, 0)
        return res_colors

    def forward(self, xyt):
        # xyt N * 3
        # xy in [0,1]
        # t is frame idx

        x = xyt[:, 0:1]
        y = xyt[:, 1:2]
        t = xyt[:, 2:3] / self.frames_num
        yx_idx = torch.cat([y, x], dim=-1)
        yx_feautres = self.plane_sample(self.yx_plane, yx_idx)

        xt_idx = torch.cat([x, t], dim=-1)
        xt_features = self.plane_sample(self.xt_plane, xt_idx)

        yt_idx = torch.cat([y, t], dim=-1)
        yt_features = self.plane_sample(self.yt_plane, yt_idx)

        F = yx_feautres * xt_features * yt_features  # [batch_size, F]

        prepared_features = self.feature_mlp(F)
        yx_pos_enc = self.positional_encoding(torch.cat([y, x], dim=-1), self.uv_pos_enc_level)
        #         print("yx_pos_enc", yx_pos_enc)
        t_pos_enc = self.positional_encoding(t, self.t_pos_enc_level)
        #         print("t_pos_enc", t_pos_enc)
        full_input = torch.cat([prepared_features, yx_pos_enc, t_pos_enc], dim=-1)
        colors = self.block2(full_input)
        return colors
