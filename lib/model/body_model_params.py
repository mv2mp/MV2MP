from typing import Dict

import jaxtyping as jt
import torch
import torch.nn as nn


class BodyModelParams(nn.Module):
    def __init__(self, num_frames: int, model_type: str = "smpl"):
        super(BodyModelParams, self).__init__()
        self.num_frames = num_frames
        self.model_type = model_type
        self.params_dim = {"betas": 10, "global_orient": 3, "transl": 3}

        if model_type == "smpl":
            self.params_dim.update({"body_pose": 69})
        else:
            assert NotImplementedError(f"Unknown model type {model_type}, exiting!")

        self.param_names = list(self.params_dim.keys())

        for param_name in self.param_names:
            if param_name == "betas":
                param = nn.Embedding(1, self.params_dim[param_name])
                param.weight.data.fill_(0)
                param.weight.requires_grad = True
                setattr(self, param_name, param)
            else:
                param = nn.Embedding(num_frames, self.params_dim[param_name])
                param.weight.data.fill_(0)
                param.weight.requires_grad = True
                setattr(self, param_name, param)

    def init_parameters(self, param_name: str, timestep: int, data: torch.Tensor, requires_grad: bool = False):
        parameter: nn.Embedding = getattr(self, param_name)

        if param_name == "betas" and self.model_type == "smpl":
            parameter.weight.data[0, : self.params_dim[param_name]] = data[: self.params_dim[param_name]]
        else:
            parameter.weight.data[timestep, : self.params_dim[param_name]] = data[: self.params_dim[param_name]]
        parameter.weight.requires_grad = requires_grad

    def set_requires_grad(self, param_name: str, requires_grad: bool = True):
        parameter: nn.Embedding = getattr(self, param_name)
        parameter.weight.requires_grad = requires_grad

    def forward(self, frame_ids: jt.Int[torch.Tensor, "N"]) -> Dict[str, torch.Tensor]:
        params = {}
        for param_name in self.param_names:
            param: nn.Embedding = getattr(self, param_name)
            if param_name == "betas" and self.model_type == "smpl":
                params[param_name] = param(torch.zeros_like(frame_ids))
            else:
                params[param_name] = param(frame_ids)
        return params
