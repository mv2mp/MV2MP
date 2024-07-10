import functools
import logging
import os
from typing import Any, Dict, Iterator, List, Literal, Optional, Protocol, Sequence
import warnings

import attrs
import jaxtyping as jt
import numpy as np

import training_registry as registry #TODO

from . import objcentric_dataset_yt_free as objcentric_dataset
from . import camera
from . import cached_ff_yt_free as cached_ff
from . import types_yt_free as types
from . import utils


logger = logging.getLogger(__package__)


@functools.lru_cache(None)
def get_vertex_based_mask(
    data: types.SingleInstanceVertexViewData,
    k_neighbors_to_fill: int = 30,
    dilate_size: int = 5,
) -> jt.Bool[np.ndarray, "H W"]:
    projected_points = camera.project_points_to_camera_np(
        K=data.intrinsics,
        Rt=data.extrinsics,
        points_3d=data.vertices,
    )

    h, w = data.image.shape[:2]
    mask = utils.get_point_based_mask(
        points=projected_points,
        image_wh=(w, h),
        k_neighbors_to_fill=k_neighbors_to_fill,
        dilate=dilate_size,
    )

    return mask


class VertexViewDataIteratorV1(Protocol):
    def get_iterator(self) -> Iterator[types.VertexViewData]:
        pass

@attrs.define(auto_attribs=True)
class CachedFFGeneratorConfig(registry.BaseConfig):
    @attrs.define(auto_attribs=True)
    class UnpackConfig(cached_ff.UnpackConfig, registry.BaseConfig.TypedContainer):
        ...

    path: str
    local_smpl_model_path: str
    frames: List[int]
    cameras: List[str]

    permute_cameras: bool
    permute_frames: Literal["none", "once", "per_camera"]

    instance_unpack_configs: Dict[str, UnpackConfig]

    object_source: objcentric_dataset.ObjectSource

    extrinsics_translation_from_mm_to_m: bool = True

    # for now those below are common for the whole dataset, but can be trivially added to the `unpack configs`
    # if for example we'd like to have different sets of meshes for the same instance
    # can come in handy for e.g. mesh metrics

    resize_ratio: float = 1.0

    generator_caching: bool = True
    save_locally: bool = False
    _target: type = attrs.field(default=attrs.Factory(lambda: CachedFFGenerator))


@registry.singleton.register
class CachedFFGenerator(VertexViewDataIteratorV1):
    ConfigType = CachedFFGeneratorConfig

    def __init__(self, config: CachedFFGeneratorConfig):
        self._config = config

        in_memory_ff = objcentric_dataset.get_ff_cache(
            path=config.path,
            local_smpl_model_path=config.local_smpl_model_path,
            object_source=self._config.object_source,
            frame_indices=tuple(config.frames),
            save_in_cache=self._config.save_locally,
            extrinsics_translation_from_mm_to_m=self._config.extrinsics_translation_from_mm_to_m,
        )

        self.dataset = cached_ff.FFCacheDataset(
            ff_cache=in_memory_ff.cache, unpack_configs=self._config.instance_unpack_configs
        )

        self._iterator_cache: Dict[Any, types.VertexViewData] = {}

    @property
    def instances(self) -> List[int]:
        warnings.warn(".instances is preserved for compatibility reasons, do not use", DeprecationWarning)

        assert len(self._config.instance_unpack_configs) == 1
        instance_unpack_config = next(iter(self._config.instance_unpack_configs.values()))
        return instance_unpack_config.taken_instances

    def get(self, camera: str, frame_id: int, unpack_key: Optional[str] = None) -> types.VertexViewData:
        """
        Args:
            unpack_key: key to select a specific unpacking configuration from the `instance_unpack_configs`
                if None, the first one is used, checked if there is only one
        """

        cache_key = camera, frame_id, unpack_key
        cache = self._iterator_cache

        if self._config.generator_caching and cache_key in cache:
            payload = cache[cache_key]
        else:
            payload = self.dataset.get(camera, frame_id, unpack_key).resize(self._config.resize_ratio)
            if self._config.generator_caching:
                cache[cache_key] = payload

        return payload

    def get_iterator(self, unpack_key: Optional[str] = None) -> Iterator[types.VertexViewData]:
        """
        Args:
            unpack_key: key to select a specific unpacking configuration from the `instance_unpack_configs`
                if None, the first one is used, checked if there is only one
        """
        assert self._config.permute_frames in ("none", "once", "per_camera")

        if self._config.permute_cameras:
            cameras_schedule: Sequence[str] = np.random.permutation(self._config.cameras)
        else:
            cameras_schedule = self._config.cameras

        if self._config.permute_frames != "none":
            frames: Sequence[int] = np.random.permutation(self.dataset.available_frames)
        else:
            frames = self.dataset.available_frames

        if self._config.permute_frames != "per_camera":
            frame_schedules_per_camera: List[Sequence[int]] = [frames for _ in self._config.cameras]
        else:
            frame_schedules_per_camera = [np.random.permutation(frames) for _ in self._config.cameras]

        taken_count = 0

        while taken_count < len(self._config.cameras) * len(frames):
            camera_idx = taken_count % len(self._config.cameras)
            frame_idx = taken_count // len(self._config.cameras)

            taken_cam = cameras_schedule[camera_idx]
            frame_schedule = frame_schedules_per_camera[camera_idx]
            taken_frame = frame_schedule[frame_idx]

            taken_count += 1
            yield self.get(taken_cam, taken_frame, unpack_key)
