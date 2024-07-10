import datetime
import enum
import functools
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Protocol, ClassVar
import warnings

import torch.utils.tensorboard.writer as tb


class Context:
    """
    nirvana dl wrapper, which also gives access to:
        - tensorboard,
        - pulsar
    """

    rank: ClassVar[str] = int(os.environ.get("LOCAL_RANK", "0"))
    world_size: ClassVar[str] = int(os.environ.get("WORLD_SIZE", "1"))
    device: ClassVar[str] = f"cuda:{rank}"

    input_data_dir: ClassVar[str] = os.environ.get("INPUT_DATA_PATH", os.getcwd())
    revision: ClassVar[str] = os.environ.get("REVISION", "REVISION_NOT_SET")

    on_nirvana = False

    def __init__(self, model_name: str, launch_file: str, comment: str = "", data_revision: str = ""):
        self.step = 0
        self.epoch = 0
        self.model_name = model_name
        self.comment = comment
        self.data_revision = data_revision or "dummy"

        launch_time = datetime.datetime.now().isoformat()
        fallback_dir = os.path.join(os.getcwd(), launch_time)
        Path(fallback_dir).mkdir(exist_ok=True, parents=True)
        print(fallback_dir)
        if any(k not in os.environ for k in ["LOGS_PATH", "DATA_PATH", "STATE_PATH"]):
            os.makedirs(fallback_dir, exist_ok=True)

        # override class-defined paths with timestamped folders
        self.log_dir = fallback_dir #os.environ.get("LOGS_PATH", fallback_dir)
        self.tboard_dir = fallback_dir #os.environ.get("LOGS_PATH", fallback_dir)
        self.data_dir = os.environ.get("DATA_PATH", fallback_dir)
        self.state_dir = fallback_dir #os.environ.get("STATE_PATH", fallback_dir)
        self.ctx_json_path = f"{self.state_dir}/ctx.json"

        self.tboard_writer: Optional[tb.SummaryWriter] = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        stdout_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        stdout_handler.setFormatter(formatter)
        self.logger.addHandler(stdout_handler)

        self.logger.info(f"rank: {self.rank}")

    @property
    def tboard(self) -> tb.SummaryWriter:
        if self.rank != 0:
            raise Exception("trying to get tboard from non-root process")
        if self.tboard_writer is None:
            self.tboard_writer = tb.SummaryWriter(log_dir=self.tboard_dir)
        return self.tboard_writer

    def save(self):
        if self.rank != 0:
            return

        with open(self.ctx_json_path, "w") as f:
            json.dump({"step": self.step, "epoch": self.epoch}, f)

    def load(self, obj: Dict[str, Any]):
        self.step = obj.get("step", 0)
        self.epoch = obj.get("epoch", 0)


def rank_zero_only(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        if Context.rank == 0:
            f(*args, **kwargs)

    return wrapped


class ScalarLoggerProtocol(Protocol):
    def __call__(self, scalars: Dict[str, float], step: Optional[int] = None, push: bool = True):
        pass


def create_scalar_logger(
    ctx: Context,
) -> ScalarLoggerProtocol:
    def split_metric_key(key: str) -> Tuple[str, List[str]]:
        parts = key.split("/")
        return parts[0], sorted(set(parts[1:]))

    def filter_scalars_from_nan_and_warn(scalars: Dict[str, float]) -> Dict[str, float]:
        filtered_scalars = {}
        for k, v in scalars.items():
            if math.isfinite(v):
                filtered_scalars[k] = v
            else:
                ctx.logger.warning(f"non-finite value in metric: {k}")
        return filtered_scalars

    @rank_zero_only
    def log_scalars(scalars: Dict[str, float], step: Optional[int] = None, push: bool = True):
        step = step if step else ctx.step
        for key, value in scalars.items():
            ctx.tboard.add_scalar(f"{key}", value, global_step=step)
            ctx.logger.info(f"step: {step}: {key}: {value}")

    return log_scalars


def create_scalar_logger_with_preservation(
    ctx: Context,
    mode: Callable[[float, float], float] = max,
) -> ScalarLoggerProtocol:
    mode_name = mode.__name__

    log_stateless = create_scalar_logger(ctx)
    preserved_values: Dict[str, float] = {}

    @rank_zero_only
    def log_scalars(scalars: Dict[str, float], step: Optional[int] = None, push: bool = True):
        log_stateless(scalars, light=push, step=step)

        for k in scalars:
            if k in preserved_values:
                preserved_values[k] = mode(scalars[k], preserved_values[k])
            else:
                preserved_values[k] = scalars[k]

        log_stateless({f"{mode_name}_{k}": v for k, v in preserved_values.items()}, push=push, step=step)

    return log_scalars


singleton_ctx: Optional[Context] = None
singleton_scalar_logger: Optional[ScalarLoggerProtocol] = None


def create_context(
    model_name: str,
    launch_file: str,
    comment: str,
    data_revision: str,
) -> Context:
    global singleton_ctx

    if singleton_ctx is not None:
        raise RuntimeError("context was already created")

    singleton_ctx = Context(
        model_name=model_name,
        launch_file=launch_file,
        comment=comment,
        data_revision=data_revision,
    )

    return singleton_ctx


def _warn_absent_context_and_create() -> Context:
    global singleton_ctx
    warnings.warn("no context was created")
    singleton_ctx = create_context(
        model_name="vr_dummy_model",
        launch_file=__package__,
        comment="",
        data_revision="",
    )
    return singleton_ctx


def get_context() -> Context:
    global singleton_ctx
    if singleton_ctx is None:
        return _warn_absent_context_and_create()
    return singleton_ctx


def get_tboard() -> tb.SummaryWriter:
    return get_context().tboard


def get_scalar_logger() -> ScalarLoggerProtocol:
    global singleton_scalar_logger
    if singleton_scalar_logger is None:
        singleton_scalar_logger = create_scalar_logger(
            get_context(),
        )
    return singleton_scalar_logger
