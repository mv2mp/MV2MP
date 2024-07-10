import abc
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import attr
import jaxtyping as jt
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from . import types

Numeric = Union[float, int]
# jt.Num[torch.Tensor, "0"] to be able to use them as losses


def psnr(
    first: jt.Float[torch.Tensor, "*N C"],
    second: jt.Float[torch.Tensor, "*N C"],
    data_range: Tuple[float, float] = (0.0, 1.0),
) -> float:
    assert data_range[0] < data_range[1]

    mse = torch.mean((first - second) ** 2)
    data_span = data_range[1] - data_range[0]
    return 10.0 * torch.log10(data_span**2 / (mse + 1e-10)).item()


class Metric(abc.ABC):
    def __init__(self, name: str, is_differentiable: bool):
        self._is_differentiable = is_differentiable
        self._name = name

    @property
    def is_differentiable(self) -> bool:
        return self._is_differentiable

    def _warn_ignore(self):
        warnings.warn(f"{type(self).__name__} is skipped")

    @abc.abstractmethod
    def __call__(self, *, prediction: Any, gt: Optional[Any] = None) -> Dict[str, Numeric]:
        pass


class ImagePSNR(Metric):
    def __init__(
        self,
        *,
        name: str = "image_psnr",
        data_range: Tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__(is_differentiable=True, name=name)
        self._data_range = data_range

    def __call__(self, *, prediction: types.Image, gt: types.Image) -> Dict[str, Numeric]:
        # these checks should be moved to decorator that checks types in runtime
        checks = [
            isinstance(prediction, types.Image),
            isinstance(gt, types.Image),
        ]

        if not all(checks):
            self._warn_ignore()
            return {}

        return {f"{self._name}": psnr(prediction.image_data, gt.image_data, self._data_range)}


class MaskedImagePSNR(Metric):
    class GT(types.Image, types.HardMasks):
        ...

    def __init__(
        self,
        name: str = "masked_image_psnr",
        data_range: Tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__(is_differentiable=True, name=name)
        self._data_range = data_range

    def __call__(
        self,
        prediction: types.Image,
        gt: GT,
    ) -> Dict[str, Numeric]:
        # these checks should be moved to decorator that checks types in runtime
        checks = [
            isinstance(prediction, types.Image),
            isinstance(gt, types.Image),
            isinstance(gt, types.HardMasks),
        ]

        if not all(checks):
            self._warn_ignore()
            return {}

        metrics = {}
        for name, mask in gt.hard_masks.items():
            assert mask.shape == gt.image_data.shape[:2]
            assert mask.shape == prediction.image_data.shape[:2]

            selected_gt_colors: jt.Float[torch.Tensor, "N C"] = gt.image_data[mask]
            selected_pred_colors = prediction.image_data[mask]

            metrics[f"{self._name}/mask={name}"] = psnr(
                selected_gt_colors,
                selected_pred_colors,
                data_range=self._data_range,
            )

        return metrics


class SSIM(Metric):
    def __init__(
        self,
        name: str = "image_ssim",
        data_range: Tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__(is_differentiable=True, name=name)
        self._data_range = data_range

    @staticmethod
    def _pt2np(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    def __call__(self, *, prediction: types.Image, gt: types.Image) -> Dict[str, Numeric]:
        # these checks should be moved to decorator that checks types in runtime
        checks = [
            isinstance(prediction, types.Image),
            isinstance(gt, types.Image),
        ]

        if not all(checks):
            self._warn_ignore()
            return {}

        np_pred = self._pt2np(prediction.image_data)
        np_gt = self._pt2np(gt.image_data)

        data_range = self._data_range[1] - self._data_range[0]
        metric = ssim(np_gt, np_pred, channel_axis=2, data_range=data_range)
        return {f"{self._name}": float(metric)}