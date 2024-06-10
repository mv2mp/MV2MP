from typing import Optional, Tuple

import jaxtyping as jt

import torch
import torch.nn.functional as F
from pytorch3d import ops

from reconstruction.vid2avatar.lib.model.smpl import SMPLServer


class SMPLDeformer(torch.nn.Module):
    def __init__(self, max_dist=0.1, K=1, gender="female", betas=None):
        super().__init__()

        self.max_dist = max_dist
        self.K = K
        self.smpl = SMPLServer(gender=gender)

        smpl_params_canoical = self.smpl.param_canonical.clone()
        smpl_params_canoical[:, 76:] = torch.tensor(betas).float().to(self.smpl.param_canonical.device)

        cano_scale, cano_transl, cano_thetas, cano_betas = torch.split(smpl_params_canoical, [1, 3, 72, 10], dim=1)
        with torch.no_grad():
            smpl_output = self.smpl(cano_scale, cano_transl, cano_thetas, cano_betas)

        self.smpl_verts: jt.Float[torch, "1 6890 3"] = torch.nn.Parameter(
            smpl_output["smpl_verts"], requires_grad=False
        )
        self.smpl_weights: jt.Float[torch.Tensor, "1 V J"] = torch.nn.Parameter(
            smpl_output["smpl_weights"], requires_grad=False
        )

    def forward(
        self,
        x: jt.Float[torch.Tensor, "N 3"],
        smpl_tfs: jt.Float[torch.Tensor, "1 J 4 4"],
        return_weights: bool = True,
        inverse: bool = False,
        smpl_verts: Optional[jt.Float[torch.Tensor, "1 6890 3"]] = None,
    ):
        if x.shape[0] == 0:
            return x

        smpl_verts = smpl_verts if smpl_verts is not None else self.smpl_verts
        weights, outlier_mask = self.query_skinning_weights_smpl_multi(
            x[None], smpl_verts=smpl_verts, smpl_weights=self.smpl_weights
        )

        if return_weights:
            return weights

        x_transformed = skinning(x.unsqueeze(0), weights, smpl_tfs, inverse=inverse)

        return x_transformed.squeeze(0), outlier_mask.squeeze(0)

    def forward_skinning(
        self,
        xc: jt.Float[torch.Tensor, "1 N 3"],
        smpl_tfs: jt.Float[torch.Tensor, "1 J 4 4"],
        smpl_verts: Optional[jt.Float[torch.Tensor, "1 6890 3"]] = None,
    ) -> Tuple[jt.Float[torch.Tensor, "1 N J"], jt.Bool[torch.Tensor, "1 N"]]:
        smpl_verts = smpl_verts if smpl_verts is not None else self.smpl_verts
        weights, outlier_mask = self.query_skinning_weights_smpl_multi(
            xc, smpl_verts=smpl_verts, smpl_weights=self.smpl_weights
        )
        x_transformed = skinning(xc, weights, smpl_tfs, inverse=False)

        return x_transformed, outlier_mask

    def reverse_skinning(
        self,
        xposed: jt.Float[torch.Tensor, "1 N 3"],
        smpl_tfs: jt.Float[torch.Tensor, "1 J 4 4"],
        smpl_verts: Optional[jt.Float[torch.Tensor, "1 6890 3"]] = None,
    ) -> Tuple[jt.Float[torch.Tensor, "1 N J"], jt.Bool[torch.Tensor, "1 N"]]:
        smpl_verts = smpl_verts if smpl_verts is not None else self.smpl_verts
        weights, outlier_mask = self.query_skinning_weights_smpl_multi(
            xposed, smpl_verts=smpl_verts, smpl_weights=self.smpl_weights
        )
        x_transformed = skinning(xposed, weights, smpl_tfs, inverse=True)
        return x_transformed, outlier_mask

    def query_skinning_weights_smpl_multi(
        self,
        pts: jt.Float[torch.Tensor, "1 N 3"],
        smpl_verts: jt.Float[torch.Tensor, "1 V 3"],
        smpl_weights: jt.Float[torch.Tensor, "1 V J"],
    ) -> Tuple[jt.Float[torch.Tensor, "1 N J"], jt.Bool[torch.Tensor, "1 N"]]:
        knn_pack = ops.knn_points(pts, smpl_verts, K=self.K, return_nn=False)

        distance_batch = knn_pack.dists
        index_batch = knn_pack.idx

        distance_batch = torch.clamp(distance_batch, max=4)
        weights_conf = torch.exp(-distance_batch)
        distance_batch = torch.sqrt(distance_batch)
        weights_conf = weights_conf / weights_conf.sum(-1, keepdim=True)
        index_batch = index_batch[0]
        weights = smpl_weights[:, index_batch, :]
        weights = torch.sum(weights * weights_conf.unsqueeze(-1), dim=-2).detach()

        outlier_mask = distance_batch[..., 0] > self.max_dist
        return weights, outlier_mask

    def query_weights(
        self,
        xc: jt.Float[torch.Tensor, "N 3"],
    ) -> jt.Float[torch.Tensor, "1 N J"]:
        weights = self.forward(xc, None, return_weights=True, inverse=False)
        return weights

    def forward_skinning_normal(self, xc, normal, cond, tfs, inverse=False):
        if normal.ndim == 2:
            normal = normal.unsqueeze(0)
        w = self.query_weights(xc[0])

        p_h = F.pad(normal, (0, 1), value=0)

        if inverse:
            # p:num_point, n:num_bone, i,j: num_dim+1
            tf_w = torch.einsum("bpn,bnij->bpij", w.double(), tfs.double())
            p_h = torch.einsum("bpij,bpj->bpi", tf_w.inverse(), p_h.double()).float()
        else:
            p_h = torch.einsum("bpn, bnij, bpj->bpi", w.double(), tfs.double(), p_h.double()).float()

        return p_h[:, :, :3]


def skinning(
    x: jt.Float[torch.Tensor, "B N D"],
    w: jt.Float[torch.Tensor, "B N J"],
    tfs: jt.Float[torch.Tensor, "B J D+1 D+1"],
    inverse: bool = False,
) -> jt.Float[torch.Tensor, "B N D"]:
    """Linear blend skinning
    Args:
        x (tensor): canonical points. shape: [B, N, D]
        w (tensor): conditional input. [B, N, J]
        tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
    Returns:
        x (tensor): skinned points. shape: [B, N, D]
    """
    x_h = F.pad(x, (0, 1), value=1.0)
    point_dim: int = x.shape[-1]
    if inverse:
        # p:n_point, n:n_bone, i,k: n_dim+1
        w_tf = torch.einsum("bpn,bnij->bpij", w, tfs)
        x_h = torch.einsum("bpij,bpj->bpi", w_tf.inverse(), x_h)
    else:
        x_h = torch.einsum("bpn,bnij,bpj->bpi", w, tfs, x_h)

    return x_h[:, :, :point_dim]
