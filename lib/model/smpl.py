from typing import Optional

import torch
import numpy as np

from reconstruction.vid2avatar.lib.smpl import V2A_SMPL_ROOT
from reconstruction.vid2avatar.lib.smpl.body_models import SMPL


class SMPLServer(torch.nn.Module):
    """
    this server implements additional translation correction for SMPL model
    which:
        - breaks naively passed HI4D parameters, i.e. everything works just fine without these corrections
        - if one wants to turn axes, particular attention should be paid to the fact, that
        even if you pass zero translation into the model, zero'th joint is not in the origin
            - if working in original axes, provided correction fixes this issue, i.e. zero'th joint is in the origin
            - if some axes are somehow turned (e.g. by 180 degrees), this correction is incorrect due to the reason above

    in order to waste time on debugging this, just write the status on different smpl_data:
        - unchanged axes, HI4D parameters: works fine without correction being applied, breaks with corrections
        - changes axes, HI4D parameters: provided vertices diverge from predicted ones, both with and without correction
            way to make it work:
                1) switch off correction
                2) do "-j0 + j0 * [1, -1, -1]", there are also more members in correction, though they cancel each other out
        - unchnaged axes, SMPL-PROTO-EMC parameters: everthing works without correction
    """

    def __init__(self, gender="neutral", betas=None, v_template=None):
        super().__init__()

        self.smpl = SMPL(
            model_path=f"{V2A_SMPL_ROOT}/smpl_model",
            gender=gender,
            batch_size=1,
            use_hands=False,
            use_feet_keypoints=False,
            dtype=torch.float32,
        )

        self.bone_parents = self.smpl.bone_parents.astype(int)
        self.bone_parents[0] = -1
        self.bone_ids = []
        self.faces = self.smpl.faces
        for i in range(24):
            self.bone_ids.append([self.bone_parents[i], i])

        self.v_template: Optional[torch.Tensor]
        if v_template is not None:
            self.register_buffer("v_template", torch.tensor(v_template).float())
        else:
            self.v_template = None

        # define the canonical pose
        param_canonical = torch.zeros((1, 86), dtype=torch.float32)
        param_canonical[0, 0] = 1
        param_canonical[0, 9] = np.pi / 6
        param_canonical[0, 12] = -np.pi / 6
        if betas is not None and self.v_template is None:
            param_canonical[0, -10:] = torch.tensor(betas).float()

        self.param_canonical: torch.Tensor
        self.register_buffer("param_canonical", param_canonical)

        self._cano_output = None

    @property
    def _lazy_canonical_output(self):
        if self._cano_output is None or self.param_canonical.device != self._cano_output["smpl_verts"].device:
            self._cano_output = self.forward(*torch.split(self.param_canonical, [1, 3, 72, 10], dim=1), absolute=True)
        return self._cano_output

    @property
    def verts_c(self) -> torch.Tensor:
        return self._lazy_canonical_output["smpl_verts"]

    @property
    def joints_c(self) -> torch.Tensor:
        return self._lazy_canonical_output["smpl_jnts"]

    @property
    def tfs_c_inv(self) -> torch.Tensor:
        return self._lazy_canonical_output["smpl_tfs"].squeeze(0).inverse()

    def forward(self, scale, transl, thetas, betas, absolute=False):
        """return SMPL output from params
        Args:
            scale : scale factor. shape: [B, 1]
            transl: global smpl pose translation. shape: [B, 3]
            thetas: pose. First 3 params - global smpl pose rotation. shape: [B, 72]
            betas: shape. shape: [B, 10]
            absolute (bool): if true return smpl_tfs wrt thetas=0. else wrt thetas=thetas_canonical.
        Returns:
            smpl_verts: vertices. shape: [B, 6893. 3]
            smpl_tfs: bone transformations. shape: [B, 24, 4, 4]
            smpl_jnts: joint positions. shape: [B, 25, 3]
        """

        # ignore betas if v_template is provided
        if self.v_template is not None:
            betas = torch.zeros_like(betas)

        smpl_output = self.smpl.forward(
            betas=betas,
            transl=torch.zeros_like(transl),
            body_pose=thetas[:, 3:],
            global_orient=thetas[:, :3],
            return_verts=True,
            return_full_pose=True,
            v_template=self.v_template,
        )

        verts = smpl_output.vertices.clone()
        joints = smpl_output.joints.clone()
        tf_mats = smpl_output.T.clone()
        tf_mats_scaled = smpl_output.T.clone()

        j0 = joints[:, 0]
        global_orient_rot = tf_mats[:, 0, :3, :3]

        corrected_translation = transl + torch.einsum("bij,bj->bi", global_orient_rot, j0) - j0

        output = {}
        output["smpl_verts"] = verts * scale.unsqueeze(1) + corrected_translation.unsqueeze(1) * scale.unsqueeze(1)
        output["smpl_jnts"] = joints * scale.unsqueeze(1) + corrected_translation.unsqueeze(1) * scale.unsqueeze(1)

        tf_mats_scaled[:, :, :3, :] = tf_mats_scaled[:, :, :3, :] * scale[:, None, None]
        tf_mats_scaled[:, :, :3, 3] = tf_mats_scaled[:, :, :3, 3] + corrected_translation[:, None] * scale[:, None]

        if not absolute:
            tf_mats_scaled = torch.einsum("bnij,njk->bnik", tf_mats_scaled, self.tfs_c_inv)

        output["smpl_tfs"] = tf_mats_scaled
        output["smpl_weights"] = smpl_output.weights
        return output
