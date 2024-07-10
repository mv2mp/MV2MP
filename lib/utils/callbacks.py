import collections
import copy
import logging
import os
import pickle
import re
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple, Union

import cv2
import jaxtyping as jt
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import torch
import torch.utils.data as td
import tqdm
import trimesh
import typing_extensions as te
from frames_folder import helpers as ffh
from frames_folder.artifact_types.vr import (
    IMAGES_UNDISTORTED,
    MESH_LIST,
    RIG_INFO_ADJUSTED,
    SEGMENTATION_MASK,
    SEGMENTATION_UNDISTORTED_MASK,
    SMPL_MODELS,
    get_artifact_type_by_name,
)
from frames_folder.frames_folder import Frame, FramesFolder
from frames_folder.table_utils import collect_frames_table_meta_set
from proto.images_pb2 import Images
from proto.mesh_pb2 import MeshList
from reconstruction.datasets.serialization import SmplModel, encode_to_proto_mesh, encode_to_proto_smpl_models
from reconstruction.helpers import training_context
from reconstruction.neuralbody.utils import accumulate_values_from_coo, np2pt, tensor_as_image
from reconstruction.smpl_fitting.tracks_postprocessing import get_world_to_smpl
from reconstruction.vid2avatar import v2a_model
from reconstruction.vid2avatar.lib.model.smpl import SMPLServer
from reconstruction.vid2avatar.lib.utils import utils
from reconstruction.vid2avatar.lib.utils.mesh_postprocessing import remove_hooves
from util.serialization import ImageFormat, serialize_image_pb

from util import yt as util_yt

logger = logging.getLogger(__name__)


def pack_meshes_to_mesh_list(meshes: List[trimesh.Trimesh]) -> MeshList:
    meshlist_proto = MeshList()

    for mesh in meshes:
        proto_mesh = encode_to_proto_mesh(np2pt(mesh.vertices), np2pt(mesh.faces))
        meshlist_proto.meshes.append(proto_mesh)

    return meshlist_proto


def pack_images_to_proto(images: Dict[str, jt.UInt8[np.ndarray, "H W *C"]]) -> Images:
    images_proto = Images()

    for key, mask in images.items():
        image_proto = serialize_image_pb(mask, format=ImageFormat.PNG_8BIT)
        images_proto.images[key].CopyFrom(image_proto)

    return images_proto


def _get_ff_interframe_step_for_metaset(frame_ids: Sequence[int]) -> int:
    possible_interframe_steps = [j - i for i, j in zip(frame_ids, frame_ids[1:])]
    if len(set(possible_interframe_steps)) == 1:
        interframe_step = possible_interframe_steps[0]
    else:
        interframe_step = 1
    return interframe_step


def set_ff_artifact_frames_meta_set(frames_folder: FramesFolder, out_artifact, interframe_step):
    frames_folder.set_artifact_frames_meta_set(
        artifact=out_artifact,
        frames_meta_set=collect_frames_table_meta_set(
            util_yt.create_client(),
            f"{frames_folder.path}/{out_artifact.path()}",
            max_frame_id_step_size=interframe_step,
        ),
    )


def save_meshes_to_ff(
    ff_path: str,
    expected_frame_ids: Sequence[int],
    meshes: Dict[int, Dict[int, trimesh.Trimesh]],
    artifact_name: str = MESH_LIST.name(),
):
    ff = ffh.from_abs_path(ff_path)
    out_artifact = get_artifact_type_by_name(artifact_name)

    frame_ids = sorted(set(expected_frame_ids).intersection(meshes.keys()))
    interframe_step = _get_ff_interframe_step_for_metaset(frame_ids)

    def generate_frames():
        metas = [meta for meta in ff.get_artifact_frames_meta(RIG_INFO_ADJUSTED) if meta.frame_id in frame_ids]
        for frame in ff.read_frames(artifacts=[RIG_INFO_ADJUSTED], frames=metas):
            meshes_for_frame = meshes[frame.frame_id]
            proto = pack_meshes_to_mesh_list(list(meshes_for_frame.values()))
            frame.set_artifact(out_artifact, proto)
            yield frame

    ff.write_frames(
        tqdm.tqdm(generate_frames(), desc="packing meshes to meshlist"),
        artifacts=[out_artifact],
        overwrite=True,
    )

    set_ff_artifact_frames_meta_set(ff, out_artifact, interframe_step)


def save_images_to_ff(
    ff_path: str,
    expected_frame_ids: Sequence[int],
    images: Iterable[Tuple[int, Union[Dict[str, jt.UInt8[np.ndarray, "H W *C"]]]]],
    artifact_name: str = SEGMENTATION_MASK.name(),
):
    ff = ffh.from_abs_path(ff_path)
    out_artifact = get_artifact_type_by_name(artifact_name)
    visited_frame_ids = set()

    def generate_frames():
        metas = [meta for meta in ff.get_artifact_frames_meta(RIG_INFO_ADJUSTED) if meta.frame_id in expected_frame_ids]
        for frame_id, images_for_frame in images:
            single_meta = [m for m in metas if m.frame_id == frame_id]
            assert len(single_meta) == 1, f"expected single meta for frame_id got {len(single_meta)}"

            visited_frame_ids.add(frame_id)

            frame: Frame = next(iter(ff.read_frames(artifacts=[RIG_INFO_ADJUSTED], frames=single_meta)))
            proto = pack_images_to_proto(images_for_frame)
            frame.set_artifact(out_artifact, proto)
            yield frame

    ff.write_frames(
        tqdm.tqdm(generate_frames(), desc="packing images to ff"),
        artifacts=[out_artifact],
        overwrite=True,
    )

    interframe_step = _get_ff_interframe_step_for_metaset(sorted(visited_frame_ids))
    set_ff_artifact_frames_meta_set(ff, out_artifact, interframe_step)


def create_smpl_model_from_params_pack(smpl_toolkit_pack, server: SMPLServer):
    scale, full_pose, smpl_shape, translation = smpl_toolkit_pack
    rotation, pose = full_pose[:, :3], full_pose[:, 3:]
    smpl_outputs = server(scale, translation, full_pose, smpl_shape)
    return SmplModel(
        poses=pose,
        shapes=smpl_shape,
        Rh=rotation,
        Th=translation,
        points=smpl_outputs["smpl_verts"][0],
        world_to_local=get_world_to_smpl(rotation.cpu().numpy(), translation.cpu().numpy()),
        model_name=server.smpl.gender,
    )


@torch.no_grad()
def get_smpl_mesh(
    smpl_parameters: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    smpl_server: SMPLServer,
) -> trimesh.Trimesh:
    scale, full_pose, smpl_shape, smpl_trans = smpl_parameters
    smpl_outputs = smpl_server.forward(scale, smpl_trans, full_pose, smpl_shape)
    smpl_vertices = smpl_outputs["smpl_verts"][0].cpu().numpy()
    return trimesh.Trimesh(vertices=smpl_vertices, faces=smpl_server.faces)


@torch.no_grad()
def save_smpl_to_ff(
    ff_path: str,
    smpl_parameters: Dict[int, Dict[int, Any]],
    pl_model: v2a_model.V2AModel,
    artifact_name: str = SMPL_MODELS.with_variant("optimized_v2a").name(),
):
    ff = ffh.from_abs_path(ff_path)
    frame_ids = sorted(smpl_parameters.keys())
    out_artifact = get_artifact_type_by_name(artifact_name)

    def generate_frames():
        metas = [meta for meta in ff.get_artifact_frames_meta(RIG_INFO_ADJUSTED) if meta.frame_id in frame_ids]
        frames = ff.read_frames(artifacts=[RIG_INFO_ADJUSTED], frames=metas)
        for frame in frames:
            assert frame.frame_id in smpl_parameters, f"frame_id {frame.frame_id} is not in smpl_parameters"
            smpls = {
                str(person_id): create_smpl_model_from_params_pack(
                    smpl_toolkit_pack, pl_model.model.smpl_server_list[person_id]
                )
                for person_id, smpl_toolkit_pack in smpl_parameters[frame.frame_id].items()
            }
            frame.set_artifact(out_artifact, encode_to_proto_smpl_models(smpls))
            yield frame

    ff.write_frames(tqdm.tqdm(generate_frames(), desc="fill smpl proto"), artifacts=[out_artifact], overwrite=True)
    set_ff_artifact_frames_meta_set(ff, out_artifact, _get_ff_interframe_step_for_metaset(frame_ids))


class ArtifactsExportToFFCallback(plc.Callback):
    DEFORMED_MESH_REGEX = r"deformed-mesh=person=(\d+)-ts=(\d+).ply"

    def __init__(self, opt, ff_path: str, dataloader: td.DataLoader):
        super().__init__()
        self._ff_path = ff_path
        self._dataloader = dataloader

        self._mask_boundary_size = opt.mask_boundary_size
        self._export_renders = opt.export_renders
        self._export_meshes = opt.export_meshes
        self._export_masks = opt.export_masks
        self._export_smpl = opt.export_smpl
        self._remove_mesh_hooves = opt.remove_mesh_hooves

        self._extracted_artifact_sfx = opt.extracted_artifact_sfx
        assert self._extracted_artifact_sfx is not None, "extracted_artifact_sfx is not set"

    @classmethod
    def get_frame_id_and_instance_from_mesh_path(
        cls,
        mesh_path: str,
        offset: int = 0,
        step: int = 1,
    ) -> Tuple[te.Annotated[int, "frame_id"], te.Annotated[int, "instance_id"]]:
        """
        Extracts frame_id and instance_id from a mesh path.
        regex_pattern is metnioned in ArtifactsExportToFFCallback
        """

        iid, ts = re.findall(cls.DEFORMED_MESH_REGEX, mesh_path)[0]
        return int(ts) * step + offset, int(iid)

    @torch.no_grad()
    def _render_image(self, batch, pl_module: v2a_model.V2AModel):
        inputs, targets = batch
        inputs = copy.deepcopy(inputs)
        pl_module.update_pose_shape_trans(inputs)

        h, w = [i.item() for i in inputs["img_size"]]
        total_pixels: int = len(inputs["uv"][0])

        pixel_per_batch: int = targets["pixel_per_batch"].item()

        render = torch.zeros((h, w, 3), dtype=torch.float32, device=inputs["uv"].device)

        chunks = utils.split_input(inputs, total_pixels, n_pixels=min(pixel_per_batch, total_pixels))
        for chunk in chunks:
            out = pl_module.model(chunk)
            render = accumulate_values_from_coo(out["rgb_values"], chunk["uv"][0].long().flip(dims=[1]), render)

        return tensor_as_image(render, "HWC")

    def render_generator(
        self, dataloader: td.DataLoader, pl_module: v2a_model.V2AModel
    ) -> Iterator[Tuple[int, Dict[str, jt.UInt8[np.ndarray, "H W C"]]]]:
        frame_id_to_check, ret = None, None
        for batch in tqdm.tqdm(dataloader, "generating renders"):
            batch = self._batch_to_device(batch, pl_module.device)
            inputs, _ = batch
            frame_id: int = inputs["ff_frame_id"].item()
            cam_name: str = inputs["camera_name"][0]

            if frame_id != frame_id_to_check:
                if frame_id_to_check is not None:
                    yield (frame_id, ret)

                frame_id_to_check = frame_id
                ret = {}

            ret[cam_name] = self._render_image(batch, pl_module)

    def mask_generator(
        self, dataloader: td.DataLoader, pl_module: v2a_model.V2AModel, meshes: Dict[int, Dict[int, trimesh.Trimesh]]
    ) -> Iterator[Tuple[int, Dict[str, jt.UInt8[np.ndarray, "H W"]]]]:
        frame_id_to_check, ret = None, None

        for batch in tqdm.tqdm(dataloader, "generating masks"):
            inputs, _ = self._batch_to_device(batch, pl_module.device)
            oh, ow = [i.item() for i in inputs["original_img_size"]]

            frame_id: int = inputs["ff_frame_id"].item()
            cam_name: str = inputs["camera_name"][0]
            mesh_list_local = list(meshes[frame_id].values())

            if frame_id != frame_id_to_check:
                if frame_id_to_check is not None:
                    yield (frame_id, ret)

                frame_id_to_check = frame_id
                ret = {}

            mask = pl_module.get_meshbased_mask(
                batch,
                boundary_size=self._mask_boundary_size,
                vertices=[np2pt(mesh.vertices, device=pl_module.device) for mesh in mesh_list_local],
                faces=[np2pt(mesh.faces, device=pl_module.device) for mesh in mesh_list_local],
            )

            mask = tensor_as_image(mask, layout="HW", clip=True)
            if mask.shape[:2] != (oh, ow):
                mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)

            ret[cam_name] = mask

        # retuen last pack
        yield (frame_id, ret)

    @staticmethod
    def _batch_to_device(batch, device):
        for pack in batch:
            for k, v in pack.items():
                if isinstance(v, torch.Tensor):
                    pack[k] = v.to(device)
                if isinstance(v, list):
                    pack[k] = [i.to(device) if isinstance(i, torch.Tensor) else i for i in v]
        return batch

    def _do_export(self, dataloader: td.DataLoader, pl_module: v2a_model.V2AModel):
        device = pl_module.device

        meshes: Dict[int, Dict[int, trimesh.Trimesh]] = collections.defaultdict(dict)
        optimized_smpl_parameters: Dict[int, Dict[int, Any]] = collections.defaultdict(dict)

        expected_frame_ids = set()
        # spin dataloader once to obtain all frame ids
        for batch in tqdm.tqdm(dataloader, "obtaining frame ids"):
            inputs, _ = batch
            expected_frame_ids.add(inputs["ff_frame_id"].item())

        ordered_frame_ids = sorted(expected_frame_ids)
        logger.info(f"found expected frame ids: {ordered_frame_ids}")

        smpl_export_visited_frame_ids = set()
        if self._remove_mesh_hooves or self._export_smpl:
            for batch in tqdm.tqdm(dataloader, "extracting smpl parameters"):
                batch = self._batch_to_device(batch, device)
                inputs, _ = batch
                timestep: int = inputs["ff_frame_id"].item()
                for person_idx in range(pl_module.persons_count):
                    if timestep not in smpl_export_visited_frame_ids:
                        smpl_toolkit_pack = pl_module.get_smpl_params_debug(inputs, person_idx, from_model=True)
                        optimized_smpl_parameters[timestep][person_idx] = smpl_toolkit_pack

                        smpl_export_visited_frame_ids.add(timestep)

        mesh_export_visited_frame_ids = set()
        if self._export_meshes or self._export_masks:
            for batch in tqdm.tqdm(dataloader, "generating meshes"):
                batch = self._batch_to_device(batch, device)
                inputs, _ = batch
                timestep: int = inputs["ff_frame_id"].item()
                for person_idx in range(pl_module.persons_count):
                    if timestep not in mesh_export_visited_frame_ids:
                        deformed_mesh = pl_module.get_deformed_mesh(batch, person_idx)
                        if self._remove_mesh_hooves:
                            _smpl_mesh = get_smpl_mesh(
                                optimized_smpl_parameters[timestep][person_idx],
                                pl_module.model.smpl_server_list[person_idx],
                            )
                            deformed_mesh = remove_hooves(deformed_mesh, _smpl_mesh)

                        meshes[timestep][person_idx] = deformed_mesh
                        mesh_export_visited_frame_ids.add(timestep)

        if self._export_renders:
            save_images_to_ff(
                self._ff_path,
                expected_frame_ids=ordered_frame_ids,
                images=self.render_generator(dataloader, pl_module),
                artifact_name=IMAGES_UNDISTORTED.with_variant(f"render_{self._extracted_artifact_sfx}").name(),
            )

        if self._export_masks:
            save_images_to_ff(
                self._ff_path,
                expected_frame_ids=ordered_frame_ids,
                images=self.mask_generator(dataloader, pl_module, meshes),
                artifact_name=SEGMENTATION_UNDISTORTED_MASK.with_variant(
                    f"v2a_bs{self._mask_boundary_size}_{self._extracted_artifact_sfx}"
                ).name(),
            )

        if self._export_meshes:
            meshes_artifact_name = MESH_LIST.with_variant(self._extracted_artifact_sfx).name()
            save_meshes_to_ff(
                self._ff_path,
                expected_frame_ids=ordered_frame_ids,
                meshes=meshes,
                artifact_name=meshes_artifact_name,
            )

        if self._export_smpl:
            smpl_artifact_name = SMPL_MODELS.with_variant(f"optimized_v2a_{self._extracted_artifact_sfx}").name()
            save_smpl_to_ff(
                self._ff_path,
                smpl_parameters=optimized_smpl_parameters,
                pl_model=pl_module,
                artifact_name=smpl_artifact_name,
            )

    def on_train_end(self, trainer: pl.Trainer, pl_module: v2a_model.V2AModel):
        self._do_export(self._dataloader, pl_module)

    def call(self, pl_module: v2a_model.V2AModel):
        self._do_export(self._dataloader, pl_module)


class MaskExportToFFCallback(plc.Callback):
    def __init__(self, ff_path: str, dataloader: td.DataLoader) -> None:
        super().__init__()
        self._ff_path = ff_path
        self._dataloader = dataloader

    @torch.no_grad()
    def _mask_export(self, dataloader: td.DataLoader, pl_module: v2a_model.V2AModel):
        pl_module = pl_module
        device = pl_module.device

        masks: Dict[int, Dict[str, jt.UInt8[np.ndarray, "H W"]]] = collections.defaultdict(dict)

        for batch in tqdm.tqdm(dataloader, "generating masks"):
            inputs, targets = batch
            for pack in [inputs, targets]:
                for k, v in pack.items():
                    if isinstance(v, torch.Tensor):
                        pack[k] = v.to(device)
                    if isinstance(v, list):
                        pack[k] = [i.to(device) if isinstance(i, torch.Tensor) else i for i in v]

            pl_module.eval()
            pl_module.update_pose_shape_trans(inputs)

            h, w = [i.item() for i in targets["img_size"]]
            total_pixels: int = len(inputs["uv"][0])

            pixel_per_batch: int = targets["pixel_per_batch"].item()

            assert "ff_frame_id" in inputs, f"ff_frame_id is in inputs with keys {inputs.keys()}"
            timestep: int = inputs[0]["ff_frame_id"].item()

            mask_render = torch.zeros((h, w, 1), dtype=torch.float32, device=inputs["uv"].device)

            chunks = utils.split_input(inputs, total_pixels, n_pixels=min(pixel_per_batch, total_pixels))
            for chunk in chunks:
                out = pl_module.model(chunk)

                mask_render = accumulate_values_from_coo(
                    out["acc_map"][..., None], chunk["uv"][0].long().flip(dims=[1]), mask_render
                )

            mask = tensor_as_image(mask_render[..., 0], "HW", clip=True)
            masks[timestep][inputs["camera_name"][0]] = mask
            save_images_to_ff(
                self._ff_path, masks, artifact_name=SEGMENTATION_UNDISTORTED_MASK.with_variant("v2a").name()
            )

    def on_train_end(self, trainer: pl.Trainer, pl_module: v2a_model.V2AModel):
        self._mask_export(self._dataloader, pl_module)

    def call(self, pl_module: v2a_model.V2AModel):
        self._mask_export(self._dataloader, pl_module)


class NirvanaCheckpointCallback(plc.ModelCheckpoint):
    def save_checkpoint(self, trainer: pl.Trainer):
        super().save_checkpoint(trainer)
        ctx = training_context.get_context()
        ctx.save()


class MemoryProfileCallback(plc.Callback):
    def __init__(self, save_dir: str, profile_every_n_steps: int) -> None:
        self._save_dir = save_dir
        os.makedirs(self._save_dir, exist_ok=True)

        self._profile_every_n_steps = profile_every_n_steps

        super().__init__()

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        torch.cuda.memory._record_memory_history(
            enabled=True,
            trace_alloc_max_entries=100000,
            trace_alloc_record_context=True,
        )

    def on_after_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.global_step % self._profile_every_n_steps == 0:
            with open(f"{self._save_dir}/memory-profile-{trainer.global_step}.pkl", "wb") as f:
                pickle.dump(torch.cuda.memory._snapshot(), f)
