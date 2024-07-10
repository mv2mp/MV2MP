import copy
import logging
from typing import List, Literal, Dict, Any

import jaxtyping as jt
import numpy as np
import omegaconf as oc
import torch

import training_registry as registry #TODO

from . import factory_yt_free as factory
from .utils import pt2np, np2pt
from .utils import get_world_ray_pixel_mask

from ..utils import utils
from . import utils as dataset_utils


logger = logging.getLogger(__name__)


class FFCacheWrapper(torch.utils.data.Dataset):
    SMPL_REPACKER_CACHE = {}
    GENDERS_TO_FIT = ["male", "female", "neutral"]

    FGD_UNPACK_KEY = "foreground"
    BGD_UNPACK_KEY = "background"

    def __init__(self, metainfo, split):
        split = copy.deepcopy(split)
        ff_config: factory.CachedFFGeneratorConfig = registry.get_targeted_config_from_dict(
            registry.singleton,
            oc.OmegaConf.to_container(split.ff_config, resolve=True),
        )

        assert isinstance(ff_config, factory.CachedFFGeneratorConfig)
        assert self.FGD_UNPACK_KEY in ff_config.instance_unpack_configs, "v2a needs foreground masks in ff config"

        if self.BGD_UNPACK_KEY in ff_config.instance_unpack_configs:
            logger.info("separate backgorund masks used")

        self._ff_path = ff_config.path
        self._ff_config = ff_config
        self._zero_non_foreground = split.zero_non_foreground

        if hasattr(split, "resize_ratio"):
            assert_msg = "simultaneously passing non-one resize_ratio in ff and split which is not supported"
            assert ff_config.resize_ratio == 1.0, assert_msg
            self._resize_ratio = split.resize_ratio
        else:
            self._resize_ratio = 1.0

        self._mask_source: Literal["dataset", "dataset_dilate", "aabb_vertices"] = "dataset"
        if hasattr(split, "mask_source"):
            self._mask_source = split.mask_source
            if self._mask_source == "aabb_vertices":
                assert (
                    not self._zero_non_foreground
                ), "masking out non-foreground is not supported with aabb_vertices based mask"

        self._ff_cache = factory.CachedFFGenerator(ff_config)
        self._frames_count = len(self._ff_cache.dataset.available_frames)
        self._n_images = len(self._ff_cache.dataset)

        self._num_sample: int = split.num_sample
        self._pixel_per_batch: int = split.pixel_per_batch
        self._sampling_strategy: str = "weighted"

        self._correct_translations: bool = split.correct_translations
        if self._correct_translations:
            assert "genders" in metainfo
            self._smpl_genders_canonical = metainfo.genders
        else:
            self._smpl_genders_canonical = None

        self._chosen_frame_ids: List[int] = split.chosen_frame_ids

        assert all(frame_id in self._ff_cache.dataset.available_frames for frame_id in self._chosen_frame_ids), (self._ff_cache.dataset.available_frames, self._chosen_frame_ids)

        self._keys = sorted(
            (frame_id, cam)
            for frame_id, cam in self._ff_cache.dataset.available_frame_camera_pairs
            if cam in ff_config.cameras and frame_id in self._chosen_frame_ids
        )

    def __len__(self):
        return len(self._keys)

    def _repack_smpl_params(
        self,
        frame_id: int,
        smpl_vertices: jt.Float[np.ndarray, "N 6890 3"],
        smpl_shapes: jt.Float[np.ndarray, "N 10"],
        smpl_poses: jt.Float[np.ndarray, "N 69"],
        world_to_smpl: jt.Float[np.ndarray, "N 4 4"],
    ) -> jt.Float[np.ndarray, "N 86"]:
        """
        This is crutch that is used to obtain correct smpl params. Should be replaced after.
        """
        if (self._ff_path, frame_id) in self.SMPL_REPACKER_CACHE:
            return self.SMPL_REPACKER_CACHE[(self._ff_path, frame_id)]

        scale = np.array([1.0])
        results = []

        smpl_genders_estimated = []

        for person_id, w2l in enumerate(world_to_smpl):
            global_translation, global_orient = dataset_utils.extract_pose_and_rodrigues(np.linalg.inv(w2l))
            smpl_pose = smpl_poses[person_id]
            smpl_shape = smpl_shapes[person_id]
            vertices = smpl_vertices[person_id]

            if self._correct_translations:
                from scipy.spatial.transform import Rotation
                from lib.model.smpl import SMPLServer

                smpl_scale = torch.ones((1, 1))
                w2l_rotvec = Rotation.from_matrix(w2l[:3, :3])
                smpl_thetas = torch.cat([np2pt(w2l_rotvec.inv().as_rotvec()), np2pt(smpl_pose)], dim=0)[None]
                smpl_translation = np2pt(global_translation)[None]
                smpl_betas = np2pt(smpl_shape)[None]

                optimal_mse = float("inf")
                optimal_reltol = float("inf")
                optimal_gender = None
                optimal_shift = None
                for gender in self.GENDERS_TO_FIT:
                    server = SMPLServer(gender)
                    smpl_outs = server.forward(
                        smpl_scale,
                        smpl_translation,
                        smpl_thetas,
                        smpl_betas,
                    )

                    estimated_vertices = pt2np(smpl_outs["smpl_verts"][0])
                    estimated_shift = (vertices - estimated_vertices).mean(axis=0)
                    mse_before = np.linalg.norm(estimated_shift)
                    mse_after = np.linalg.norm(vertices - (estimated_vertices + estimated_shift))

                    logger.info(
                        f"SMPL translation for frame_id={frame_id}, person_id={person_id} with gender={gender}: before {mse_before}; after {mse_after}"
                    )

                    if mse_after < optimal_mse:
                        optimal_mse = mse_after
                        optimal_gender = gender

                        optimal_reltol = mse_after / (mse_before + 1e-3)
                        optimal_shift = estimated_shift

                assert (
                    optimal_reltol < 1e-2 or optimal_mse < 1e-2
                ), f"Optimal reltol={optimal_reltol} is too high or mse={optimal_mse}"
                smpl_genders_estimated.append(optimal_gender)

                assert optimal_shift is not None
                smpl_trans = global_translation + optimal_shift
            else:
                smpl_trans = global_translation

            result = np.concatenate([scale, smpl_trans, global_orient, smpl_pose, smpl_shape], axis=0)
            results.append(result)

        if self._correct_translations:
            assert len(smpl_genders_estimated) == len(self._smpl_genders_canonical)
            assert all(
                est == true for est, true in zip(self._smpl_genders_canonical, smpl_genders_estimated)
            ), f"didn't match genders from config: {self._smpl_genders_canonical} and estimated {smpl_genders_estimated}"

        final = np.array(results)
        self.SMPL_REPACKER_CACHE[(self._ff_path, frame_id)] = final
        return final

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        frame_id, camera = self._keys[idx]

        skip = False

        data_items = {
            k: self._ff_cache.get(
                camera=camera,
                frame_id=frame_id,
                unpack_key=k,
            )
            for k in self._ff_config.instance_unpack_configs
        }

        original_data_item = data_items[self.FGD_UNPACK_KEY]
        original_img_size = original_data_item.image.shape[:2]

        data_item = original_data_item.resize(self._resize_ratio)

        if self._mask_source == "dataset":
            fuzzy_foreground_mask = data_item.all_instances_mask
        elif self._mask_source == "dataset_dilate":
            import cv2

            kernel_size = int(0.05 * data_item.image.shape[0])

            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask_selected = 255 * data_item.all_instances_mask.astype(np.uint8)
            mask_dilate = cv2.dilate(mask_selected.copy(), kernel)
            fuzzy_foreground_mask = mask_dilate > 0

        elif self._mask_source == "aabb_vertices":
            with torch.no_grad():
                fuzzy_foreground_mask = np.zeros_like(data_item.all_instances_mask)
                for instance_id in range(data_item.n_instances):
                    fuzzy_foreground_mask += pt2np(
                        get_world_ray_pixel_mask(
                            np2pt(data_item.get_vertices(instance_id)),
                            np2pt(data_item.intrinsics),
                            np2pt(data_item.extrinsics),
                            aabb_margin=0.05,
                            mask_shape=data_item.image.shape[:2],
                        )
                    )
        else:
            raise NotImplementedError

        if self.BGD_UNPACK_KEY in data_items:
            background_mask = data_items[self.BGD_UNPACK_KEY].all_instances_mask
        else:
            background_mask = np.logical_not(fuzzy_foreground_mask)

        img = data_item.image / 255.0
        if self._zero_non_foreground:
            img[~fuzzy_foreground_mask] = 0
        # for eroded masks that means that we spoil good signal here,
        # so no rgb loss should be applied in non fg-region

        extrinsics = data_item.extrinsics
        extr_4x4 = np.eye(4)
        extr_4x4[0:3, :] = extrinsics
        pose_4x4 = np.linalg.inv(extr_4x4)
        cam_pose = pose_4x4[:3, :]

        intrinsics = data_item.intrinsics

        world_to_smpl = np.stack(data_item.world_to_local, axis=0)
        world_to_smpl_final = np.eye(4)[None].repeat(world_to_smpl.shape[0], axis=0)
        world_to_smpl_final[:, :3] = world_to_smpl

        h, w = img_size = img.shape[:2]
        uv = np.mgrid[: img_size[0], : img_size[1]].astype(np.int32)
        uv = np.flip(uv, axis=0).copy().transpose(1, 2, 0).astype(np.float32)

        assert data_item.vertices is not None
        assert data_item.shapes is not None
        assert data_item.poses is not None

        smpl_params = self._repack_smpl_params(
            frame_id,
            np.stack(data_item.vertices, axis=0),
            np.stack(data_item.shapes, axis=0),
            np.stack(data_item.poses, axis=0),
            world_to_smpl_final,
        )

        if self._num_sample > 0:
            data = {
                "rgb": img,
                "uv": uv,
                "foreground_mask": fuzzy_foreground_mask,
                "background_mask": background_mask,
            }

            samples, index_outside, sample_fg_mask, sample_bg_mask = utils.weighted_sampling(
                data,
                img_size,
                self._num_sample,
            )

            inputs = {
                "uv": samples["uv"].astype(np.float32),
                "intrinsics": intrinsics.astype(np.float32),
                "extrinsics": extrinsics.astype(np.float32),
                "pose": cam_pose.astype(np.float32),
                "smpl_params": smpl_params.astype(np.float32),
                "index_outside": index_outside.astype(np.int32),
                "idx": idx,
                "camera_name": data_item.camera_name,
                "frame_num": data_item.timestep,
                "ff_frame_id": frame_id,
                "world_to_smpl": world_to_smpl,
                "skip": skip,
                "img_size": img_size,
                "original_img_size": original_img_size,
            }
            images = {
                "rgb": samples["rgb"].astype(np.float32),
                "img_size": img_size,
                "original_img_size": original_img_size,
                "sample_fg_mask": sample_fg_mask,
                "sample_bg_mask": sample_bg_mask,
                "pixel_per_batch": self._pixel_per_batch,
            }
            return inputs, images
        else:
            ys, xs = np.where(fuzzy_foreground_mask)
            whwh = np.array([w, h, w, h])
            margin = 15
            box = np.array([xs.min() - margin, ys.min() - margin, xs.max() + margin, ys.max() + margin]).clip(
                min=0, max=whwh
            )
            box_mask = utils.box_mask_from_xyxy(box, img_size)

            inputs = {
                "uv": uv[box_mask].astype(np.float32),
                "intrinsics": intrinsics.astype(np.float32),
                "extrinsics": extrinsics.astype(np.float32),
                "pose": cam_pose.astype(np.float32),
                "smpl_params": smpl_params.astype(np.float32),
                "idx": idx,
                "camera_name": data_item.camera_name,
                "frame_num": data_item.timestep,
                "ff_frame_id": frame_id,
                "world_to_smpl": world_to_smpl,
                "skip": skip,
                "img_size": img_size,
                "original_img_size": original_img_size,
            }

            images = {
                "rgb": img[box_mask].reshape(-1, 3).astype(np.float32),
                "sample_fg_mask": fuzzy_foreground_mask[box_mask].flatten().astype(bool),
                "sample_bg_mask": background_mask[box_mask].flatten().astype(bool),
                "img_size": img_size,
                "original_img_size": original_img_size,
                "pixel_per_batch": self._pixel_per_batch,
            }

            return inputs, images
