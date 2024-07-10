import copy
from typing import List, Tuple, Optional

import attrs
import cv2
import jaxtyping as jt
import numpy as np

def resize_masks(arr: jt.Bool[np.ndarray, "H W C"], wh: Tuple[int, int]) -> jt.Bool[np.ndarray, "h w C"]:
    final_bit_masks = [
        cv2.resize(b_mask.astype(np.uint8), wh, cv2.INTER_NEAREST).astype(bool) for b_mask in arr.transpose(2, 0, 1)
    ]

    return np.stack(final_bit_masks, axis=-1)

@attrs.define(auto_attribs=True)
class VertexViewData:
    """dataclass to store per camera information about objects on the scene, it includes:
    - point clouds of objects
    - images and given camera
    - transformation of point cloud to its coordinate frame
    """

    image: jt.UInt8[np.ndarray, "H W C"]

    all_instances_mask: jt.Bool[np.ndarray, "H W"]
    # this thing is always equal to not mask.sum(axis=-1),
    # as there may not be all instances in `mask`

    extrinsics: jt.Float[np.ndarray, "3 4"]
    intrinsics: jt.Float[np.ndarray, "3 3"]

    masks: List[jt.Bool[np.ndarray, "H W"]] = attrs.Factory(list)

    world_to_local: List[Optional[jt.Float[np.ndarray, "3 4"]]] = attrs.Factory(list)

    vertices: List[Optional[jt.Float[np.ndarray, "V 3"]]] = attrs.Factory(list)
    faces: List[Optional[jt.Int[np.ndarray, "F 3"]]] = attrs.Factory(list)

    shapes: List[Optional[jt.Float[np.ndarray, "S"]]] = attrs.Factory(list)
    poses: List[Optional[jt.Float[np.ndarray, "P"]]] = attrs.Factory(list)

    camera_name: str = ""
    timestep: int = 0

    @property
    def n_instances(self) -> int:
        return len(self.vertices)

    def get_instance_mask(self, instance_id: int) -> jt.Bool[np.ndarray, "H W"]:
        """returns part of the orginal mask, corresponding to passed instance copy, not slice"""
        assert self.masks
        assert instance_id < self.n_instances
        return self.masks[instance_id].copy()

    def get_other_instances_mask(self, instance_id: int) -> jt.Bool[np.ndarray, "H W"]:
        return self.all_instances_mask ^ self.get_instance_mask(instance_id)

    def get_vertices(self, instance_id: int) -> Optional[jt.Float[np.ndarray, "R 3"]]:
        if self.vertices is None:
            return None
        return self.vertices[instance_id]

    def get_faces(self, instance_id: int) -> Optional[jt.Int[np.ndarray, "F 3"]]:
        if self.faces is None:
            return None
        return self.faces[instance_id]

    def get_pose(self, instance_id: int) -> Optional[jt.Float[np.ndarray, "P"]]:
        if self.poses is None:
            return None
        return self.poses[instance_id]

    def get_shape(self, instance_id: int) -> Optional[jt.Float[np.ndarray, "S"]]:
        if self.shapes is None:
            return None
        return self.shapes[instance_id]

    def get_world_to_local(self, instance_id: int) -> Optional[jt.Float[np.ndarray, "3 4"]]:
        if self.world_to_local is None:
            return None
        return self.world_to_local[instance_id]

    def resize(self, resize_ratio: float) -> "VertexViewData":
        """
        returns copy with resized:
         - image
         - mask
         - intrinsics
        """
        data_copy = copy.deepcopy(self)
        if resize_ratio == 1.0:
            return data_copy

        h, w = data_copy.image.shape[:2]

        new_w = int(resize_ratio * w)
        new_h = int(resize_ratio * h)

        data_copy.intrinsics[0] *= new_w / w
        data_copy.intrinsics[1] *= new_h / h
        data_copy.image = cv2.resize(data_copy.image, (new_w, new_h), cv2.INTER_AREA)

        for idx, mask in enumerate(data_copy.masks):
            data_copy.masks[idx] = resize_masks(mask[..., None], (new_w, new_h))[..., 0]

        all_instances_mask = resize_masks(data_copy.all_instances_mask[..., None], (new_w, new_h))
        data_copy.all_instances_mask = all_instances_mask[..., 0]
        return data_copy
    

@attrs.define
class SingleInstanceVertexViewData:
    """this dataclass stores single instance data,
    its usage is discouraged in favor of VertexViewData,
    which has handy methods to work with multiple instances

    rationale for this 'do not use' is the fact that one may want to inherit from VertexViewData,
    in such case one will need to implement its 'single instance' counterpart
    """

    image: jt.UInt8[np.ndarray, "H W C"]
    mask: jt.Bool[np.ndarray, "H W 2"]
    other_instances_mask: jt.Bool[np.ndarray, "H W 2"]
    extrinsics: jt.Float[np.ndarray, "3 4"]
    intrinsics: jt.Float[np.ndarray, "3 3"]
    vertices: jt.Float[np.ndarray, "V 3"]
    world_to_local: jt.Float[np.ndarray, "3 4"]

    faces: Optional[jt.Int[np.ndarray, "F 3"]] = None

    camera_name: str = ""
    timestep: int = 0

    # TODO: maybe this index doesn't belong here, should be on processing side
    # after ablation one can consider inheriting this class as subclass of VertexViewData
    instance_index: int = 0

    def __hash__(self) -> int:
        return hash((self.camera_name, self.timestep, self.instance_index))

    def __eq__(self, other: "SingleInstanceVertexViewData"):
        return (
            type(self) == type(other)
            and other.camera_name == self.camera_name
            and other.timestep == self.timestep
            and other.instance_index == self.instance_index
        )

    @classmethod
    def from_vertex_view_data(
        cls,
        data: VertexViewData,
        instance_id: int,
        instance_index: int,
    ) -> "SingleInstanceVertexViewData":
        """generate single-instance data from multi-instance one

        Args:
            data (VertexViewData): multi-instance data
            instance_id (int): instance id in multi-instance data
            instance_index (int): ordered sequence number to use in modeling; is in range [0, num_instances_to_model)

        Returns:
            SingleInstanceVertexViewData: single-instance data
        """
        return cls(
            image=data.image,
            mask=data.get_instance_mask(instance_id=instance_id),
            other_instances_mask=data.get_other_instances_mask(instance_id=instance_id),
            extrinsics=data.extrinsics,
            intrinsics=data.intrinsics,
            vertices=data.get_vertices(instance_id=instance_id),
            faces=data.get_faces(instance_id=instance_id),
            world_to_local=data.get_world_to_local(instance_id=instance_id),
            camera_name=data.camera_name,
            timestep=data.timestep,
            instance_index=instance_index,
        )
