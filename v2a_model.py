import os
import copy
import logging
from typing import List, Sequence, Tuple

import attrs
import cv2
import jaxtyping as jt

import mvrec_metrics
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import trimesh
import tqdm


from reconstruction.helpers import training_context as context
from reconstruction.helpers import mesh_visualization
from reconstruction.neuralbody.utils import accumulate_values_from_coo, tensor_as_image, np2pt, pt2np
from reconstruction.neuralbody.datasets.utils import create_boundary_on_mask

from reconstruction.vid2avatar.lib.model.v2a import V2A
from reconstruction.vid2avatar.lib.model.body_model_params import BodyModelParams
from reconstruction.vid2avatar.lib.model.deformer import SMPLDeformer
from reconstruction.vid2avatar.lib.model.loss import Loss

from reconstruction.vid2avatar.lib.utils.meshing import generate_mesh

from reconstruction.vid2avatar.lib.model.deformer import skinning
from reconstruction.vid2avatar.lib.model.smpl import SMPLServer
from reconstruction.vid2avatar.lib.utils import utils


logger = logging.getLogger(__name__)


def squeeze_dict(d):
    keys = d.keys()
    for key in keys:
        if isinstance(d[key], int):
            d[key] = torch.tensor([d[key]])
            continue
        d[key] = torch.tensor(d[key][None])
    return d


@attrs.define(auto_attribs=True)
class TestStepGT(
    mvrec_metrics.types.Image,
    mvrec_metrics.types.HardMasks,
):
    ...


@attrs.define(auto_attribs=True)
class TestStepPred(mvrec_metrics.types.Image):
    ...


METRIC_LIST: List[mvrec_metrics.metrics.Metric] = [
    mvrec_metrics.metrics.SSIM(data_range=(0, 1)),
    mvrec_metrics.metrics.MaskedImagePSNR(data_range=(0, 1)),
    mvrec_metrics.metrics.ImagePSNR(data_range=(0, 1)),
]


class V2AModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.validation_step_outputs = []

        self.opt = opt
        num_training_frames: int = opt.dataset.metainfo.frames_count
        self.genders: str = opt.dataset.metainfo.genders
        self.persons_count: int = self.opt.model.persons_count
        self.training_indices = list(range(0, num_training_frames))
        self.update_mesh_every_epoch = self.opt.model.update_mesh_every_epoch
        self.body_model_lr_coeff = self.opt.model.body_model_lr_coeff

        self.body_model_params_list: Sequence[BodyModelParams] = nn.ModuleList()
        self.body_model_params_initialized_masks: Sequence[jt.Bool[torch.Tensor, "N"]] = nn.ParameterList()

        for _ in range(self.persons_count):
            self.body_model_params_list.append(BodyModelParams(num_training_frames, model_type="smpl"))

            all_uninitialized = nn.Parameter(torch.zeros(num_training_frames, dtype=torch.bool), requires_grad=False)
            self.body_model_params_initialized_masks.append(all_uninitialized)

        betas_list = []
        for person_id in range(self.persons_count):
            body_params: BodyModelParams = self.body_model_params_list[person_id]
            body_params_dict = body_params.forward(torch.tensor([0]))
            betas_list.append(body_params_dict["betas"].clone().detach().requires_grad_(False))

        self.model: V2A = V2A(
            opt.model,
            num_training_frames,
            opt.dataset.base_ff_config.frames,
            opt.dataset.metainfo.all_cameras,
            betas_list,
            self.genders,
        )
        self.loss = Loss(opt.model.loss, self.persons_count)

    def configure_optimizers(self):
        params = [{"params": self.model.parameters(), "lr": self.opt.model.learning_rate}]
        for body_model_param in self.body_model_params_list:
            # append the parameters of each BodyModelParams with a lower learning rate
            params.append(
                {
                    "params": body_model_param.parameters(),
                    "lr": self.opt.model.learning_rate * self.body_model_lr_coeff,
                }
            )

        self.optimizer = torch.optim.Adam(params, lr=self.opt.model.learning_rate, eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.opt.model.sched_milestones, gamma=self.opt.model.sched_factor
        )
        return [self.optimizer], [self.scheduler]

    def update_pose_shape_trans(self, inputs):
        frame_num = inputs["frame_num"]
        smpl_pose = []
        smpl_shape = []
        smpl_trans = []
        for person_id in range(self.persons_count):
            person_body_model_params: BodyModelParams = self.body_model_params_list[person_id](frame_num)
            smpl_pose.append(
                torch.cat((person_body_model_params["global_orient"], person_body_model_params["body_pose"]), dim=1)
            )
            smpl_shape.append(person_body_model_params["betas"])
            smpl_trans.append(person_body_model_params["transl"])

        inputs["smpl_pose"] = torch.cat(smpl_pose)[None]
        inputs["smpl_shape"] = torch.cat(smpl_shape)[None]
        inputs["smpl_trans"] = torch.cat(smpl_trans)[None]
        inputs["current_epoch"] = self.current_epoch

    def _is_timestep_initialized(self, frame_num: int) -> bool:
        return all(
            self.body_model_params_initialized_masks[person_id][frame_num] for person_id in range(self.persons_count)
        )

    def training_step(self, batch):
        inputs, targets = batch
        frame_num: int = inputs["frame_num"]
        if inputs["skip"]:
            logger.info(f"skipping frame num {frame_num}")
            return None

        assert len(inputs["smpl_params"]) == 1, "only batch_size=1 is supported"

        is_initialized = self._is_timestep_initialized(frame_num)

        if not is_initialized:
            assert any(
                not self.body_model_params_initialized_masks[person_id][frame_num]
                for person_id in range(self.persons_count)
            ), "some body model params are initialized, while some are not"

            smpl_params = inputs["smpl_params"][0]

            for person_id in range(self.persons_count):
                _, smpl_trans, global_orient, smpl_pose, smpl_shape = torch.split(
                    smpl_params[person_id],
                    [1, 3, 3, 69, 10],
                    dim=0,
                )

                param_dict = {
                    "betas": smpl_shape,
                    "global_orient": global_orient,
                    "body_pose": smpl_pose,
                    "transl": smpl_trans,
                }

                person_body_model_params: BodyModelParams = self.body_model_params_list[person_id]
                per_person_mask = self.body_model_params_initialized_masks[person_id]
                for param_name in person_body_model_params.param_names:
                    assert param_name in param_dict, f"param_name {param_name} not in param_dict"

                    per_person_mask[frame_num] = True
                    person_body_model_params.init_parameters(
                        param_name=param_name,
                        timestep=frame_num,
                        data=param_dict[param_name],
                        requires_grad=True,
                    )

        self.update_pose_shape_trans(inputs)
        model_outputs = self.model.forward(inputs)
        loss_output = self.loss(model_outputs, targets)

        for k, v in loss_output.items():
            self.log(k, v.item(), prog_bar=True, on_step=True)
        self.check_for_nan()
        loss = loss_output["loss"]
        return loss

    def check_for_nan(self):
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                raise ValueError(f"NaN value encountered in {name}. Terminating training.")

    def on_train_epoch_end(self) -> None:
        # Canonical mesh update every self.update_mesh_every_epoch epochs
        if self.current_epoch != 0 and self.current_epoch % self.update_mesh_every_epoch == 0:
            for person_id in range(self.persons_count):
                cond = {"smpl": torch.zeros(1, 69).float().cuda()}
                mesh_canonical = generate_mesh(
                    lambda x: self.query_oc(person_id, x, cond),
                    self.model.smpl_server_list[person_id].verts_c[0],
                    point_batch=10000,
                    res_up=2,
                )
                self.model.set_mesh_vf_cano(
                    person_number=person_id,
                    v=torch.tensor(
                        mesh_canonical.vertices[None],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    f=torch.tensor(
                        mesh_canonical.faces,
                        dtype=torch.int64,
                        device=self.device,
                    ),
                )

        return super().on_train_epoch_end()

    def query_oc(self, person_id, x, cond):
        x = x.reshape(-1, 3)
        mnfld_pred = self.model.implicit_networks[person_id](x, cond)[:, :, 0].reshape(-1, 1)
        return {"sdf": mnfld_pred}

    def query_wc(self, person_id, x):
        x = x.reshape(-1, 3)
        w = self.model.deformers[person_id].query_weights(x)

        return w

    def query_od(self, person_id, x, cond, smpl_tfs, smpl_verts):
        x = x.reshape(-1, 3)
        x_c, _ = self.model.deformers[person_id].forward(
            x, smpl_tfs, return_weights=False, inverse=True, smpl_verts=smpl_verts
        )
        output = self.model.implicit_networks[person_id](x_c, cond)[0]
        sdf = output[:, 0:1]

        return {"sdf": sdf}

    def get_smpl_params_debug(self, inputs, person_id, from_model: bool = True):
        """
        Usecase:
            server = model.model.smpl_server_list[person_id]
            smpl_toolkit_pack = model.get_smpl_params_debug(inputs, person_id, from_model=True)
            scale, full_pose, smpl_shape, smpl_trans = smpl_toolkit_pack

            outs = server.forward(scale, smpl_trans, full_pose, smpl_shape)
            # e.g. get vertices:
            verts = pt2np(outs['smpl_verts'])[0]
        Returns:
            smpl_params: Tuple[.....]
        """
        inputs = copy.deepcopy(inputs)
        if from_model:
            self.update_pose_shape_trans(inputs)
            params = self.model.get_inputs(inputs, person_id)
            params = (params[0][None], *params[1:])
        else:
            _smpl_params = inputs["smpl_params"][0]
            scale, smpl_trans, global_orient, smpl_pose, smpl_shape = torch.split(
                _smpl_params[person_id],
                [1, 3, 3, 69, 10],
                dim=0,
            )

            full_pose = torch.cat((global_orient, smpl_pose), dim=0)
            params = scale[None], full_pose[None], smpl_shape[None], smpl_trans[None]
        return params

    @torch.no_grad()
    def get_canonical_mesh(self, batch, person_id: int) -> trimesh.Trimesh:
        inputs, _ = batch

        inputs = copy.deepcopy(inputs)
        self.update_pose_shape_trans(inputs)

        smpl_params = self.body_model_params_list[person_id].forward(torch.tensor([0], device=self.device))
        smpl_server = SMPLServer(
            gender=self.model.genders[person_id],
            betas=smpl_params["betas"].clone().detach().requires_grad_(False),
        ).to(self.device)

        cond = {"smpl": inputs["smpl_pose"][:, person_id, 3:] / np.pi}
        mesh_canonical = generate_mesh(
            lambda x: self.query_oc(person_id, x, cond),
            smpl_server.verts_c[0].to(self.device),
            point_batch=10000,
            res_up=3,
        )

        mesh_canonical = trimesh.Trimesh(mesh_canonical.vertices, mesh_canonical.faces, process=False)
        return mesh_canonical

    @torch.no_grad()
    def get_deformed_mesh(self, batch, person_id: int) -> trimesh.Trimesh:
        inputs, _ = batch

        self.update_pose_shape_trans(inputs)
        scale, smpl_pose, smpl_shape, smpl_trans = self.model.get_inputs(inputs, person_id)

        mesh_canonical = self.get_canonical_mesh(batch, person_id)

        smpl_outputs = self.model.smpl_server_list[person_id](scale, smpl_trans, smpl_pose, smpl_shape)
        smpl_tfs = smpl_outputs["smpl_tfs"]

        verts_cano = np2pt(mesh_canonical.vertices, device=scale.device)

        smpl_params = self.body_model_params_list[person_id].forward(torch.tensor([0], device=self.device))
        smpl_deformer = SMPLDeformer(
            gender=self.model.genders[person_id],
            betas=smpl_params["betas"].clone().detach().requires_grad_(False),
        ).to(self.device)

        weights = smpl_deformer.query_weights(verts_cano)
        verts_deformed = skinning(verts_cano[None], weights, smpl_tfs)
        return trimesh.Trimesh(vertices=pt2np(verts_deformed[0]), faces=mesh_canonical.faces, process=False)

    @torch.no_grad()
    def get_meshbased_mask(
        self,
        batch,
        vertices: List[jt.Float[torch.Tensor, "V 3"]],
        faces: List[jt.Int[torch.Tensor, "F 3"]],
        boundary_size: int = 10,
    ) -> jt.Float[torch.Tensor, "H W"]:
        inputs, targets = batch

        modified_inputs = copy.deepcopy(inputs)

        hw: Tuple[int, int] = tuple([i.item() for i in inputs["img_size"]])

        boundary_masks = []
        base_masks_pt = []
        for v, f in zip(vertices, faces):
            base_mask: jt.Float[torch.Tensor, "N H W C"] = mesh_visualization.render_mesh_as_mask(
                vertices=v,
                faces=f,
                intrinsics=modified_inputs["intrinsics"],
                extrinsics=modified_inputs["extrinsics"],
                img_size_hw=hw,
            )

            base_mask_single_channel: jt.Float[torch.Tensor, "N H W"] = base_mask[0, ..., 0]
            base_masks_pt.append(base_mask_single_channel)

            bool_mask_hw = pt2np(base_mask_single_channel > 0.5)
            boundary_mask = create_boundary_on_mask(mask=bool_mask_hw, dilate_kernel_size=boundary_size)

            boundary_masks.append(boundary_mask)

        boundary_mask: np.ndarray = np.any(np.stack(boundary_masks, axis=0), axis=0)
        ys, xs = np.where(boundary_mask)
        uv_samples = np.stack([xs, ys], axis=1).astype(np.float32)

        modified_inputs["uv"] = torch.tensor(uv_samples, dtype=torch.float32, device=self.device)[None]
        mask_to_fill: jt.Float[torch.Tensor, "H W 1"] = torch.stack(base_masks_pt, dim=0).sum(dim=0)[..., None]
        mask_to_fill[np2pt(boundary_mask, device=mask_to_fill.device)] = 0.0

        total_pixels = len(uv_samples)
        if total_pixels == 0:
            return mask_to_fill[..., 0]

        pixel_per_batch: int = targets["pixel_per_batch"].item()

        chunks = utils.split_input(modified_inputs, total_pixels, n_pixels=min(pixel_per_batch, total_pixels))
        for chunk in tqdm.tqdm(chunks, desc="filling mask in get_meshbased_mask()"):
            out = self.model(chunk)
            mask_to_fill = accumulate_values_from_coo(
                out["acc_map"][..., None], chunk["uv"][0].long().flip(dims=[1]), mask_to_fill
            )

        mask_to_fill = mask_to_fill[..., 0]
        return mask_to_fill

    @torch.no_grad()
    def test_step(self, batch, *args):
        inputs, targets = batch

        self.model.eval()
        self.update_pose_shape_trans(inputs)

        h, w = [i.item() for i in inputs["img_size"]]
        total_pixels: int = len(inputs["uv"][0])

        pixel_per_batch: int = targets["pixel_per_batch"].item()
        timestep: int = inputs["frame_num"].item()
        camera_name: str = inputs["camera_name"][0]

        id_log_str = f"ts={timestep}/cam={camera_name}"
        id_log_str_path = id_log_str.replace("/", "_")

        for person_id in range(self.persons_count):
            id_log_str_w_person = f"{id_log_str}/instance_id={person_id}"
            id_log_str_w_person_path = id_log_str_w_person.replace("/", "_")

            mesh_canonical = self.get_canonical_mesh(batch, person_id)
            mesh_canonical.export(
                os.path.join(context.get_context().state_dir, f"{id_log_str_w_person_path}_step={self.global_step}.ply")
            )

            mesh_deformed = self.get_deformed_mesh(batch, person_id)
            mesh_deformed.export(
                os.path.join(
                    context.get_context().state_dir, f"deformed_{id_log_str_w_person_path}_step={self.global_step}.ply"
                )
            )

        render = torch.zeros((h, w, 3), dtype=torch.float32, device=inputs["uv"].device)
        mask_render = torch.zeros((h, w, 1), dtype=torch.float32, device=inputs["uv"].device)

        gt = accumulate_values_from_coo(
            targets["rgb"][0],
            inputs["uv"][0].long().flip(dims=[1]),
            torch.zeros((h, w, 3), dtype=torch.float32, device=inputs["uv"].device),
        )

        gt_mask = accumulate_values_from_coo(
            targets["sample_fg_mask"][0][..., None],
            inputs["uv"][0].long().flip(dims=[1]),
            torch.zeros((h, w, 1), dtype=torch.bool, device=inputs["uv"].device),
        )[..., 0]

        chunks = utils.split_input(inputs, total_pixels, n_pixels=min(pixel_per_batch, total_pixels))
        for chunk in tqdm.tqdm(chunks):
            out = self.model(chunk)
            for k, v in out.items():
                if isinstance(v, list) and len(v) == self.persons_count:
                    out[k] = v[person_id]
                else:
                    out[k] = v

            render = accumulate_values_from_coo(out["rgb_values"], chunk["uv"][0].long().flip(dims=[1]), render)
            mask_render = accumulate_values_from_coo(
                out["acc_map"][..., None], chunk["uv"][0].long().flip(dims=[1]), mask_render
            )

        mask_render = mask_render[..., 0]

        context.get_tboard().add_image(
            f"{id_log_str}/pred_mask",
            tensor_as_image(mask_render, "HW"),
            global_step=self.global_step,
            dataformats="HWC",
        )

        context.get_tboard().add_image(
            f"{id_log_str}/pred",
            tensor_as_image(render, "HWC"),
            global_step=self.global_step,
            dataformats="HWC",
        )

        context.get_tboard().add_image(
            f"{id_log_str}/gt",
            tensor_as_image(gt, "HWC"),
            global_step=self.global_step,
            dataformats="HWC",
        )

        cv2.imwrite(
            os.path.join(context.get_context().state_dir, f"render_{id_log_str_path}_step={self.global_step}.png"),
            cv2.cvtColor(tensor_as_image(render, "HWC"), cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(context.get_context().state_dir, f"mask_render_{id_log_str_path}_step={self.global_step}.png"),
            cv2.cvtColor(tensor_as_image(mask_render, "HW"), cv2.COLOR_RGB2BGR),
        )

        prediction = TestStepPred(image_data=render)
        gt = TestStepGT(
            hard_masks={"common_mask": gt_mask},
            image_data=gt,
        )

        metric_dict = {
            f"{k}/eval_samples/{id_log_str}": v
            for metric in METRIC_LIST
            for k, v in metric(prediction=prediction, gt=gt).items()
        }

        context.get_scalar_logger()(metric_dict, step=self.global_step, push=True)

    def validation_step(self, batch, *args):
        return self.test_step(batch, *args)
