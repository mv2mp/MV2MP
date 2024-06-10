from argparse import Namespace
import os
from typing import Dict, Any, Optional, Container

import omegaconf as oc

import pytorch_lightning as pl
from pytorch_lightning.loggers import base as pl_loggers_base
import numpy as np
import torch
import torch.utils.tensorboard.writer as tb
import torch.utils.tensorboard.summary as tbs

from reconstruction.helpers import training_context as context

from pytorch_lightning.core.saving import save_hparams_to_yaml
from pytorch_lightning.utilities import _OMEGACONF_AVAILABLE


class PulsarTensorboardLogger(pl_loggers_base.LightningLoggerBase):
    NAME_HPARAMS_FILE = "hparams.yaml"

    def __init__(self):
        super().__init__()
        self.hparams = {}

    @property
    def log_dir(self) -> str:
        return context.get_context().log_dir

    @property
    def save_dir(self) -> Optional[str]:
        return context.get_context().state_dir

    @property
    def experiment(self) -> tb.SummaryWriter:
        r"""
        Actual tensorboard object. To use TensorBoard features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_tensorboard_function()

        """
        return context.get_tboard()

    @context.rank_zero_only
    def log_hyperparams(
        self,
        params: Dict[str, Namespace],
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """Record hyperparameters. TensorBoard logs with and without saved hyperparameters are incompatible, the
        hyperparameters are then not displayed in the TensorBoard. Please delete or move the previously saved logs
        to display the new ones with hyperparameters.

        Args:
            params: a dictionary-like container with the hyperparameters
            metrics: Dictionary with metric names as keys and measured quantities as values
        """

        params = self._convert_params(params)

        # store params to output
        if _OMEGACONF_AVAILABLE and isinstance(params, Container):
            self.hparams = oc.OmegaConf.merge(self.hparams, params)
        else:
            self.hparams.update(params)

        # format params into the suitable for tensorboard
        params = self._flatten_dict(params)
        params = self._sanitize_params(params)

        if metrics is None:
            metrics = {"hp_metric": -1}
        elif not isinstance(metrics, dict):
            metrics = {"hp_metric": metrics}

        if metrics:
            self.log_metrics(metrics, 0)
            exp, ssi, sei = tbs.hparams(params, metrics)
            writer: tb.FileWriter = self.experiment._get_file_writer()
            writer.add_summary(exp)
            writer.add_summary(ssi)
            writer.add_summary(sei)

    @context.rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        scalar_logger = context.get_scalar_logger()
        scalar_logger(metrics, step=step, push=True)

    @context.rank_zero_only
    def log_graph(self, model: pl.LightningModule, input_array=None):
        pass

    @context.rank_zero_only
    def save(self):
        super().save()
        hparams_file = os.path.join(self.save_dir, self.NAME_HPARAMS_FILE)
        save_hparams_to_yaml(hparams_file, self.hparams)

    @context.rank_zero_only
    def finalize(self, status: str) -> None:
        self.experiment.flush()
        self.experiment.close()
        self.save()

    @property
    def name(self) -> str:
        """Get the name of the experiment.

        Returns:
            The name of the experiment.
        """
        ctx = context.get_context()
        return f"model:{ctx.model_name}-data:{ctx.data_revision}"

    @property
    def version(self) -> int:
        """Get the experiment version.

        Returns:
            The experiment version if specified else the next version.
        """
        return 0

    @staticmethod
    def _sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
        params = pl_loggers_base.LightningLoggerBase._sanitize_params(params)
        return {k: str(v) if isinstance(v, (torch.Tensor, np.ndarray)) and v.ndim > 1 else v for k, v in params.items()}
