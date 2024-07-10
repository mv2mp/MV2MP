from typing import Dict

import attr
import jaxtyping as jt
import torch


@attr.s(auto_attribs=True)
class Image:
    image_data: jt.Float[torch.Tensor, "H W C"]

    @classmethod
    def image_from_uint8(cls, image: jt.UInt8[torch.Tensor, "H W C"]) -> "Image":
        return Image(image_data=image / 255.0)


@attr.s(auto_attribs=True)
class SparseColors:
    sparse_colors_data: jt.Float[torch.Tensor, "N C"]
    sparse_colors_sampled_locations: jt.Int[torch.Tensor, "N xy"]

    def sample_sparse_rgb_from_other_image(self, image: Image) -> "SparseColors":
        color_values = image.image_data[
            self.sparse_colors_sampled_locations[:, 1], self.sparse_colors_sampled_locations[:, 0]
        ]

        return SparseColors(
            sparse_colors_data=color_values, sparse_colors_sampled_locations=self.sparse_colors_sampled_locations
        )


@attr.s(auto_attribs=True)
class SparseHardMask:
    sparse_presence_data: jt.Bool[torch.Tensor, "N"]
    sparse_presence_sampled_locations: jt.Int[torch.Tensor, "N xy"]


@attr.s(auto_attribs=True)
class SparseHardMasks:
    sparse_presence_data: Dict[str, SparseHardMask]

    def add_sparse_hard_mask(
        self,
        name: str,
        data: jt.Bool[torch.Tensor, "N"],
        locations: jt.Int[torch.Tensor, "N xy"],
    ):
        assert name not in self.sparse_presence_data

        self.sparse_presence_data[name] = SparseHardMask(
            sparse_presence_data=data,
            sparse_presence_sampled_locations=locations,
        )


@attr.s(auto_attribs=True)
class HardMasks:
    hard_masks: Dict[str, jt.Bool[torch.Tensor, "H W"]]

    def add_hard_mask(self, name: str, mask: jt.Bool[torch.Tensor, "H W"]):
        assert name not in self.hard_masks
        self.hard_masks[name] = mask


@attr.s(auto_attribs=True)
class SoftMasks:
    soft_masks: Dict[str, jt.Float[torch.Tensor, "H W"]]

    def add_soft_mask(self, name: str, mask: jt.Float[torch.Tensor, "H W"]):
        assert name not in self.soft_masks
        self.soft_masks[name] = mask