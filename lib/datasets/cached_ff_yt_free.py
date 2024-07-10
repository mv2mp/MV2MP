import copy
import math
from typing import Callable, Dict, List, Sequence, Tuple, Optional
import warnings

import attrs
import jaxtyping as jt
import numpy as np
import typing_extensions as te

from . import objcentric_dataset_yt_free as objcentric_dataset
from . import types_yt_free as types
from . import utils


def vertex_view_data_from_ocf(
    src: objcentric_dataset.ObjectCentricFrame,
    mask_key: Optional[str],
    per_instance_colors: Sequence[int],
    seqno_from_frame_id: Callable[[int], int] = lambda x: x,
    segmentation_mask_bitwise: bool = False,
    all_instances_segmentation_mask_threshold: int = 0,
    taken_instances: Optional[Sequence[int]] = None,
) -> types.VertexViewData:
    """
    Args:
        src (objcentric_dataset.ObjectCentricFrame): frame to unpack
        mask_key (Optional[str]): key to access mask in frame.masks_bytes
        per_instance_colors (Sequence[int]): instance mask colors in taken mask
        seqno_from_frame_id (Callable[[int], int], optional): function to convert frame_id to seqno
        segmentation_mask_bitwise (bool, optional): flag for processing mask as bitwise
        all_instances_segmentation_mask_threshold (int, optional): threshold for all_instances_mask
            it is required for masks that store confidence instead of a constant value
        taken_instances (Optional[Sequence[int]], optional): instances to unpack, if none - take all available

    Note:
        It is expected that taken_instances is aligned with per_instance_colors,
        i.e. taken_instances[i] corresponds to per_instance_colors[i],
        although no checks for consistency is done here
    """

    intrinsics = np.copy(src.intrinsics)
    extrinsics = np.copy(src.extrinsics)[:3]

    if taken_instances is None:
        taken_instances = list(range(len(src.objects_info)))

    per_instance_masks = []
    all_instances_mask = np.ones(src.image_array.shape[:2], dtype=bool)

    if mask_key in src.masks_bytes:
        per_instance_masks = src.get_per_instance_masks(
            per_instance_colors,
            mask_key=mask_key,
            bitwise=segmentation_mask_bitwise,
        )
        all_instances_mask = src.get_all_instances_mask(
            mask_key=mask_key, threshold=all_instances_segmentation_mask_threshold
        )
    else:
        warnings.warn(f"no mask found for {mask_key} in frame={src.frame_id}, cam={src.camera_name}")

    return types.VertexViewData(
        image=utils.ensure_rgb(src.image_array),
        masks=per_instance_masks,
        all_instances_mask=all_instances_mask,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        vertices=[src.objects_info[obj_idx].vertices for obj_idx in taken_instances],
        faces=[src.objects_info[obj_idx].faces for obj_idx in taken_instances],
        shapes=[src.objects_info[obj_idx].shapes for obj_idx in taken_instances],
        poses=[src.objects_info[obj_idx].poses for obj_idx in taken_instances],
        world_to_local=[
            (
                src.objects_info[obj_idx].world_to_local[:3]
                if src.objects_info[obj_idx].world_to_local is not None
                else None
            )
            for obj_idx in taken_instances
        ],
        camera_name=src.camera_name,
        timestep=seqno_from_frame_id(src.frame_id),
    )


@attrs.define(auto_attribs=True)
class UnpackConfig:
    segmentation_mask_artifact_name: str
    taken_instances: List[int]
    mask_per_instance_colors: List[int]
    is_mask_bitwise: bool
    all_instances_mask_threshold: int = 0

    def __attrs_post_init__(self):
        if len(self.taken_instances) != len(self.mask_per_instance_colors):
            warnings.warn(
                f"""
                taken_instances={self.taken_instances} and
                mask_per_instance_colors={self.mask_per_instance_colors}
                are not aligned, is this intentional?"""
            )

        if self.is_mask_bitwise:
            assert self.all_instances_mask_threshold == 0, "all_instances_mask_threshold must be 0 for bitwise masks"
            assert all(
                math.log2(c) == int(math.log2(c)) for c in self.mask_per_instance_colors
            ), "colors must be power of 2 for bitwise masks"


class FFCacheDataset:
    def __init__(
        self,
        ff_cache: Dict[Tuple[int, str], objcentric_dataset.ObjectCentricFrame],
        unpack_configs: Dict[str, UnpackConfig],
    ):
        """
        Args:
            ff_cache (Dict[Tuple[int, str], objcentric_dataset.ObjectCentricFrame]): is supplied either with:
             - objcentric_dataset.ObjectCentricFFCache.cache
             - objcentric_dataset.ObjectCentricFFCache.load(pkl_cache_path)
            unpack_configs (Dict[str, UnpackConfig]): unpack_configs for each unpack_key
        """

        self._cache = ff_cache
        self._unpack_configs = unpack_configs

        available_frames, available_cameras = zip(*self._cache.keys())
        self._available_frames = sorted(set(available_frames))
        self._available_cameras = sorted(set(available_cameras))

        self._unique_extrinsics, self._unique_intrinsics = self._get_unique_cam_params()
        self._all_extrinsics, self._all_intrinsics = self._get_all_cam_params()

        self._frame_id_to_seqno = {frame_id: num for num, frame_id in enumerate(self.available_frames)}

    def __len__(self) -> int:
        return len(self._cache)

    @property
    def available_frame_camera_pairs(
        self,
    ) -> List[Tuple[te.Annotated[int, "frame_id"], te.Annotated[str, "camera_name"]]]:
        return list(self._cache.keys())

    @property
    def available_frames(self) -> Sequence[int]:
        return self._available_frames

    @property
    def available_cameras(self) -> Sequence[str]:
        return self._available_cameras

    @property
    def unique_extrinsics(self) -> Dict[te.Annotated[str, "camera name"], jt.Float[np.ndarray, "3 4"]]:
        return self._unique_extrinsics

    @property
    def all_extrinsics(self) -> jt.Float[np.ndarray, "3 4"]:
        return self._all_extrinsics

    @property
    def num_frames(self):
        return len(self._available_frames)

    def get(self, camera: str, frame_id: int, unpack_key: Optional[str] = None) -> types.VertexViewData:
        """
        Args:
            camera (str): camera name
            frame_id (int): frame id
            unpack_key (Optional[str]): key to unpack config, if none - checks that only one config is given, and uses it
        """
        cache_key = (frame_id, camera)
        value = copy.deepcopy(self._cache[cache_key])
        return self._vertex_view_data_from_ocf(value, self._sanitize_and_get_unpack_config(unpack_key))

    def get_multiview(
        self, cameras: List[str], frame_id: int, unpack_key: Optional[str] = None
    ) -> List[types.VertexViewData]:
        """
        Args:
            cameras (List[str]): camera names
            frame_id (int): frame id
            unpack_key (Optional[str]): key to unpack config, if none - checks that only one config is given, and uses it

        Note:
            Though vertices are shared between frames, its size is negligible wrt contained images,
            therefore we don't introduce new container for this particluar case
        """

        return [self.get(camera=camera, frame_id=frame_id, unpack_key=unpack_key) for camera in cameras]

    def _sanitize_and_get_unpack_config(self, unpack_key: Optional[str] = None) -> UnpackConfig:
        if unpack_key is None:
            assert len(self._unpack_configs) == 1, "unpack_key must be provided if there are multiple unpack_configs"
            return next(iter(self._unpack_configs.values()))
        assert unpack_key in self._unpack_configs, f"unpack_key={unpack_key} not found in unpack_configs"
        return self._unpack_configs[unpack_key]

    def _vertex_view_data_from_ocf(
        self,
        src: objcentric_dataset.ObjectCentricFrame,
        unpack_config: UnpackConfig,
    ) -> types.VertexViewData:
        assert src.frame_id in self._frame_id_to_seqno
        return vertex_view_data_from_ocf(
            src,
            mask_key=unpack_config.segmentation_mask_artifact_name,
            per_instance_colors=unpack_config.mask_per_instance_colors,
            seqno_from_frame_id=self._frame_id_to_seqno.get,
            segmentation_mask_bitwise=unpack_config.is_mask_bitwise,
            all_instances_segmentation_mask_threshold=unpack_config.all_instances_mask_threshold,
        )

    def _get_unique_cam_params(
        self,
    ) -> Tuple[
        Dict[te.Annotated[str, "camera name"], jt.Float[np.ndarray, "3 4"]],
        Dict[te.Annotated[str, "camera name"], jt.Float[np.ndarray, "3 3"]],
    ]:
        per_cam_extrinsics = {}
        per_cam_intrinsics = {}
        for (_, camera_name), frame in self._cache.items():
            if camera_name is per_cam_extrinsics:
                continue
            per_cam_extrinsics[camera_name] = frame.extrinsics
            per_cam_intrinsics[camera_name] = frame.intrinsics
        return per_cam_extrinsics, per_cam_intrinsics

    def _get_all_cam_params(
        self,
    ) -> Tuple[jt.Float[np.ndarray, "3 4"], jt.Float[np.ndarray, "3 3"]]:
        cam_extrinsics = []
        cam_intrinsics = []
        for (_, camera_name), frame in self._cache.items():
            cam_extrinsics.append(frame.extrinsics)
            cam_intrinsics.append(frame.intrinsics)
        return cam_extrinsics, cam_intrinsics
