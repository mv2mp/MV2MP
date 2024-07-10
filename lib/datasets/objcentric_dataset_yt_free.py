import enum
import functools
import glob
import io
import logging
import multiprocessing as mp
import os
import math
import pickle
import shutil
import tarfile
import threading
import trimesh
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
from pathlib import Path
import warnings

import attrs
import cv2
import jaxtyping as jt
import numpy as np
import tempfile
import typing_extensions as te
import tqdm
import zstd


logger = logging.getLogger(__name__)

@functools.lru_cache
def get_smpl_faces(local_smpl_model_path: str) -> jt.Int[np.ndarray, "N 3"]:
    smpl_dict = _get_smpl_pickle_dict(local_smpl_model_path)
    return smpl_dict["f"]

@functools.lru_cache
def _get_smpl_pickle_dict(local_smpl_model_path: str, gender: Literal["male", "female"] = 'male') -> Dict[str, Any]:
    assert gender in local_smpl_model_path.lower()

    # hack taken from https://github.com/facebookresearch/DensePose/issues/142#issuecomment-435645312
    with open(local_smpl_model_path, "rb") as f:
        raw_smpl_data = pickle._Unpickler(f)
        raw_smpl_data.encoding = "latin1"
        smpl_dict: Dict[str, Any] = raw_smpl_data.load()

    return smpl_dict

def invert_rt(Rt: np.ndarray) -> np.ndarray:
    to_inv = np.eye(4)
    to_inv[:3] = Rt
    return np.linalg.inv(to_inv)

def extract_hi4d_mesh(hi4d_mesh_tar_zst: bytes, to_dir: str):
    with tarfile.open(fileobj=io.BytesIO(zstd.decompress(hi4d_mesh_tar_zst)), mode="r") as tar:
        tar.extractall(path=to_dir)

def hi4d_Rt_from_rtvecs(
    rotvecs: jt.Float[np.ndarray, "N 3"],
    tvecs: jt.Float[np.ndarray, "N 3"],
    axis_flip: Optional[jt.Float[np.ndarray, "3"]] = None,
) -> np.ndarray:
    Rts = []

    if axis_flip is None:
        fix3 = np.ones(3)
    else:
        fix3 = axis_flip

    for rvec, tvec in zip(rotvecs, tvecs):
        R, _ = cv2.Rodrigues(rvec[None])
        R = np.diag(fix3) @ R
        tvec = tvec * fix3
        Rts.append(invert_rt(np.concatenate((R, tvec[:, None]), axis=1)))

    return np.array(Rts)

def get_hi4d_smpl_info(
    path: str,
) -> Tuple[
    te.Annotated[List[jt.Float[np.ndarray, "N 3"]], "vertices"],
    te.Annotated[List[jt.Float[np.ndarray, "3 4"]], "world to local transforms"],
    te.Annotated[List[jt.Float[np.ndarray, "S"]], "shapes"],
    te.Annotated[List[jt.Float[np.ndarray, "P"]], "poses"],
]:
    saved = np.load(path)

    fix3 = np.array([1, 1, 1])
    vertices_world = saved["verts"] * fix3
    Rt_world_to_local = hi4d_Rt_from_rtvecs(rotvecs=saved["global_orient"], tvecs=saved["transl"], axis_flip=fix3)

    return (
        [vertex_pack for vertex_pack in vertices_world],
        [Rt for Rt in Rt_world_to_local],
        [pack for pack in saved["betas"]],
        [pack for pack in saved["body_pose"]],
    )


@attrs.define(auto_attribs=True)
class ObjectInfo:
    world_to_local: Optional[jt.Float[np.ndarray, "3 4"]] = None

    vertices: Optional[jt.Float[np.ndarray, "N 3"]] = None
    faces: Optional[jt.Int[np.ndarray, "F 3"]] = None

    shapes: Optional[jt.Float[np.ndarray, "S"]] = None
    poses: Optional[jt.Float[np.ndarray, "P"]] = None

    def __attrs_post_init__(self):
        if self.faces is not None:
            assert self.vertices is not None, "Vertices must be provided when faces are provided"

class ObjectSource(str, enum.Enum):
    SMPL_HI4D = "smpl_hi4d"

def objects_info_from_ff(
    path: str,
    frame_id: int,
    local_smpl_model_path: str = "/home/flynn/arc/vr/reconstruction/vid2avatar_yt_free/smpl_models/SMPL_MALE.pkl",
    object_source: ObjectSource = ObjectSource.SMPL_HI4D,
) -> List[ObjectInfo]:
    path = Path(path)
    if object_source == ObjectSource.SMPL_HI4D:
        vertices, world_to_local, shapes, poses = get_hi4d_smpl_info(
            path/f'smpl/{frame_id:06d}.npz'
        )
        faces = [get_smpl_faces(local_smpl_model_path) for _ in vertices]
        return [
            ObjectInfo(vertices=v, faces=f, world_to_local=Rt, shapes=sh, poses=po)
            for v, f, Rt, sh, po in zip(vertices, faces, world_to_local, shapes, poses)
        ]
    else:
        raise ValueError(f"Unknown object source: {object_source}")

def decompress_image_from_buffer(compressed_buffer: bytes) -> np.ndarray:
    compressed_data = np.frombuffer(compressed_buffer, np.uint8)
    decompressed_image = cv2.imdecode(compressed_data, cv2.IMREAD_UNCHANGED)
    return decompressed_image

@attrs.define(auto_attribs=True, slots=False)
class ObjectCentricFrame:
    frame_id: int
    camera_name: str
    image_bytes: bytes
    extrinsics: jt.Float[np.ndarray, "3 4"]
    intrinsics: jt.Float[np.ndarray, "3 3"]

    masks_bytes: Dict[str, bytes] = attrs.field(default=attrs.Factory(dict))

    objects_info: List[ObjectInfo] = attrs.Factory(list)

    def __hash__(self) -> int:
        return hash(self.hash_key)

    def _sanitize_mask_key(self, mask_key: Optional[str]) -> str:
        if mask_key is None:
            warnings.warn("mask_key is not provided, it is deprecated", DeprecationWarning)
            assert len(self.masks_keys) == 1, "mask_key must be provided if there are multiple masks"
            mask_key = self.masks_keys[0]

        assert mask_key in self.masks_keys, f"mask_key {mask_key} not found in {self.masks_keys}"
        return mask_key

    @property
    def hash_key(self) -> Tuple[int, str]:
        return self.frame_id, self.camera_name

    @functools.cached_property
    def image_array(self) -> jt.UInt8[np.ndarray, "H W C"]:
        image = decompress_image_from_buffer(self.image_bytes)
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                raise NotImplementedError

        return image

    @functools.cached_property
    def mask_array(self) -> jt.UInt8[np.ndarray, "H W C"]:
        warnings.warn("mask_array is deprecated, use mask_arrays instead", DeprecationWarning)
        assert len(self.masks_keys) == 1, "mask_array works only if there is one mask"
        return self.mask_arrays[self.masks_keys[0]]

    @functools.cached_property
    def mask_arrays(self) -> Dict[str, jt.UInt8[np.ndarray, "H W C"]]:
        masks = {}

        for mask_name, mask_bytes in self.masks_bytes.items():
            mask = decompress_image_from_buffer(mask_bytes)
            if len(mask.shape) == 2:
                mask = mask[..., None]
            masks[mask_name] = mask

        return masks

    @property
    def masks_keys(self) -> List[str]:
        return list(self.masks_bytes.keys())

    def get_per_instance_masks(
        self,
        instance_colors: Sequence[int],
        mask_key: Optional[str] = None,
        bitwise: bool = False,
    ) -> List[jt.Bool[np.ndarray, "H W"]]:
        mask_key_sanitized = self._sanitize_mask_key(mask_key)

        if bitwise:
            assert all(int(math.log2(c)) == math.log2(c) for c in instance_colors), "colors must be powers of 2"

        mask: np.ndarray = self.mask_arrays[mask_key_sanitized].sum(axis=2)

        if bitwise:
            return [(mask & color) == color for color in instance_colors]
        else:
            return [mask == color for color in instance_colors]

    def get_all_instances_mask(self, mask_key: Optional[str] = None, threshold: int = 0) -> jt.Bool[np.ndarray, "H W"]:
        mask_key_sanitized = self._sanitize_mask_key(mask_key)
        mask: np.ndarray = self.mask_arrays[mask_key_sanitized]
        return (mask > threshold).sum(axis=2) != 0

    @classmethod
    def load_from_frame_id(
        cls,
        path: str,
        frame_id: int,
        local_smpl_model_path: str,
        object_source: ObjectSource = ObjectSource.SMPL_HI4D,
        extrinsics_translation_from_mm_to_m: bool = True,
    ) -> Dict[Tuple[int, str], "ObjectCentricFrame"]:
        cameras_info = np.load(f'{path}/cameras/rgb_cameras.npz')
        cameras: List[str] = cameras_info['ids'].tolist()

        path = Path(path)
        images_bytes = []
        for camera in cameras:
            images_bytes.append((path / f'images/{camera}/{frame_id:06d}.jpg').read_bytes())

        pack: Dict[Tuple[int, str], ObjectCentricFrame] = {}
        objects_info = objects_info_from_ff(path=path, frame_id=frame_id, local_smpl_model_path=local_smpl_model_path, object_source=object_source)
        for cam_id, camera in enumerate(cameras):
            packed_frame = cls(
                camera_name=camera,
                frame_id=frame_id,
                image_bytes=images_bytes[cam_id],
                masks_bytes={},
                extrinsics=cameras_info["extrinsics"][cam_id][:3],
                intrinsics=cameras_info["intrinsics"][cam_id],
                objects_info=objects_info,
            )

            pack[packed_frame.hash_key] = packed_frame

        return pack


class ObjectCentricFFCache:
    def __init__(
        self,
        path: str,
        local_smpl_model_path: str,
        object_source: ObjectSource = ObjectSource.SMPL_HI4D,
        frame_indices: Optional[List[int]] = None,
        cache_workers: int = 16,
        extrinsics_translation_from_mm_to_m: bool = True,
    ):
        self._path = Path(path)
        self._object_source = object_source

        if frame_indices is not None:
            self.frames_to_read = sorted(frame_indices)
        else:
            self.frames_to_read = sorted(int(x.stem) for x in (path/'smpl').glob("*.npz"))

        step = max(int(len(self.frames_to_read) / cache_workers), 1)
        chunks = [self.frames_to_read[start : start + step] for start in range(0, len(self.frames_to_read), step)]

        closure = functools.partial(
            self.read_from_metas,
            path=self._path,
            local_smpl_model_path=local_smpl_model_path,
            object_source=object_source,
            extrinsics_translation_from_mm_to_m=extrinsics_translation_from_mm_to_m,
        )

        with mp.Pool(len(chunks)) as pool:
            frame_chunks = pool.map(closure, chunks)

        self.cache: Dict[
            Tuple[te.Annotated[int, "frame_id"], te.Annotated[str, "camera_name"]], ObjectCentricFrame
        ] = {}

        for chunk in frame_chunks:
            self.cache.update(chunk)

    @staticmethod
    def read_from_metas(
        frame_indices: List[int],
        path: str,
        local_smpl_model_path: str,
        object_source: ObjectSource = ObjectSource.SMPL_HI4D,
        extrinsics_translation_from_mm_to_m: bool = True,
    ) -> Dict[Tuple[int, str], ObjectCentricFrame]:
        path = Path(path)
        assert path.exists()

        returned: Dict[Tuple[int, str], ObjectCentricFrame] = {}
        for frame_id in frame_indices:
            unpacked = ObjectCentricFrame.load_from_frame_id(
                path=path,
                frame_id=frame_id,
                object_source=object_source,
                local_smpl_model_path=local_smpl_model_path,
                extrinsics_translation_from_mm_to_m=extrinsics_translation_from_mm_to_m,
            )

            returned.update(unpacked)
        return returned

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "ObjectCentricFFCache":
        with open(path, "rb") as f:
            return pickle.load(f)

@functools.lru_cache
def get_ff_cache(
    path: str,
    local_smpl_model_path: str,
    object_source: ObjectSource = ObjectSource.SMPL_HI4D,
    frame_indices: Optional[Sequence[int]] = None,
    extrinsics_translation_from_mm_to_m: bool = True,
    cache_workers: int = 16,
    cache_dir: str = "~/.cache/ff-cache",
    save_in_cache: bool = False,
) -> ObjectCentricFFCache:
    import hashlib
    import pickle

    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    config_to_hash = {
        "path": path,
        "local_smpl_model_path": local_smpl_model_path,
        "object_source": object_source,
        "frame_indices": frame_indices,
        "extrinsics_translation_from_mm_to_m": extrinsics_translation_from_mm_to_m,
    }

    config_hash = hashlib.md5(str(config_to_hash).encode()).hexdigest()
    trial_path = os.path.join(cache_dir, config_hash)

    if os.path.exists(trial_path):
        with open(trial_path, "rb") as f:
            return pickle.load(f)

    ff_cache = ObjectCentricFFCache(
        path=path,
        local_smpl_model_path=local_smpl_model_path,
        object_source=object_source,
        cache_workers=cache_workers,
        frame_indices=list(frame_indices) if frame_indices is not None else None,
        extrinsics_translation_from_mm_to_m=extrinsics_translation_from_mm_to_m,
    )

    if save_in_cache:
        with open(trial_path, "wb") as f:
            pickle.dump(ff_cache, f)

    return ff_cache

if __name__ == "__main__":
    # print(objects_info_from_ff('/home/flynn/arc/vr/reconstruction/vid2avatar_yt_free/data/pair21/hug21', 11))
    out=ObjectCentricFFCache('/home/flynn/arc/vr/reconstruction/vid2avatar_yt_free/data/pair21/hug21', '/home/flynn/arc/vr/reconstruction/vid2avatar_yt_free/smpl_models/SMPL_MALE.pkl', frame_indices=[11,12,13])
    print(out.cache.keys())