import logging

import torch
from torch import nn
from functools import reduce
from itertools import combinations

logger = logging.getLogger(__name__)


class Loss(nn.Module):
    def __init__(self, opt, persons_count):
        super().__init__()
        self.eikonal_weight = opt.eikonal_weight
        self.bce_weight = opt.bce_weight
        self.opacity_sparse_weight = opt.opacity_sparse_weight
        self.in_shape_weight = opt.in_shape_weight
        self.sdf_loss_weight = opt.sdf_loss_weight
        self.dynamic_weight = opt.dynamic_weight
        self.eps = 1e-6
        # self.milestone = 200
        self.milestone = opt.milestone
        self.opacity_sparse_denom = opt.opacity_sparse_denom
        self.l1_loss = nn.L1Loss(reduction="mean")
        self.l2_loss = nn.MSELoss(reduction="mean")
        self.persons_count = persons_count
        self.filter_rgb_by_segmentation = opt.filter_rgb_by_segmentation
        self.alpha_mask_source = opt.alpha_mask_source

    # L1 reconstruction loss for RGB values
    def get_rgb_loss(self, rgb_values, rgb_gt):
        rgb_loss = self.l1_loss(rgb_values, rgb_gt)
        return rgb_loss

    # Eikonal loss introduced in IGR
    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=-1) - 1) ** 2).mean()
        return eikonal_loss

    # BCE loss for clear boundary
    def get_bce_loss(self, acc_map):
        if torch.any(acc_map < 0) or torch.any(acc_map > 1):
            logger.info(f"Tensor contains values outside the range [0, 1]: {acc_map}")

        acc_map = torch.clamp(acc_map, min=0, max=1)
        binary_loss = (
            -1 * (acc_map * (acc_map + self.eps).log() + (1 - acc_map) * (1 - acc_map + self.eps).log()).mean() * 2
        )
        return binary_loss

    # Global opacity sparseness regularization
    def get_opacity_sparse_loss(self, acc_map, index_off_surface):
        if index_off_surface.sum() == 0:
            return torch.tensor(0.0, device=acc_map.device)

        opacity_sparse_loss = self.l1_loss(acc_map[index_off_surface], torch.zeros_like(acc_map[index_off_surface]))
        return opacity_sparse_loss

    # Optional: This loss helps to stablize the training in the very beginning
    def get_in_shape_loss(self, acc_map, index_in_surface):
        if index_in_surface.sum() == 0:
            return torch.tensor(0.0, device=acc_map.device)

        in_shape_loss = self.l1_loss(acc_map[index_in_surface], torch.ones_like(acc_map[index_in_surface]))
        return in_shape_loss

    def get_sdf_loss(self, sdf_list, device):
        num_sdfs = len(sdf_list)
        if num_sdfs < 2:
            return torch.tensor(0.0, device=device)

        losses = []
        for sdf1, sdf2 in combinations(sdf_list, 2):
            negative_overlap = (sdf1 < 0) & (sdf2 < 0)
            if negative_overlap.sum() == 0:
                continue

            product_mean = torch.mean(sdf1[negative_overlap] * sdf2[negative_overlap])
            losses.append(product_mean)

        if not losses:
            return torch.tensor(0.0, device=device)

        return torch.mean(torch.stack(losses))

    def forward(self, model_outputs, ground_truth):
        nan_filter = torch.all(torch.isfinite(model_outputs["rgb_values"]), dim=1)
        rgb_gt = ground_truth["rgb"][0]

        sample_fg_mask = ground_truth["sample_fg_mask"][0]  # 1 if bodies
        sample_bg_mask = ground_truth["sample_bg_mask"][0]  # 1 if background

        rgb_filter = nan_filter
        if self.filter_rgb_by_segmentation:
            rgb_filter = rgb_filter & sample_fg_mask

        rgb_loss = self.get_rgb_loss(model_outputs["rgb_values"][rgb_filter], rgb_gt[rgb_filter])

        if self.alpha_mask_source == "segmentation":
            index_in_surface = nan_filter & sample_fg_mask
            index_off_surface = nan_filter & sample_bg_mask
        elif self.alpha_mask_source == "surface":
            index_in_surface = nan_filter & reduce(torch.logical_or, model_outputs["index_in_surface_list"])
            index_off_surface = nan_filter & reduce(torch.logical_and, model_outputs["index_off_surface_list"])
        else:
            raise ValueError(f"Unknown alpha mask source: {self.alpha_mask_source}")

        bce_loss = self.get_bce_loss(model_outputs["acc_map"])
        opacity_sparse_loss = self.get_opacity_sparse_loss(model_outputs["acc_map"], index_off_surface)
        in_shape_loss = self.get_in_shape_loss(model_outputs["acc_map"], index_in_surface)

        eikonal_loss = 0
        for person_number in range(self.persons_count):
            eikonal_loss = self.get_eikonal_loss(model_outputs["grad_theta_list"][person_number]) + eikonal_loss

        curr_epoch_for_loss = min(self.milestone, model_outputs["epoch"])  # will not increase after the milestone

        sdf_list = model_outputs["sdf_for_loss"]
        sdf_loss = self.get_sdf_loss(sdf_list, model_outputs["acc_map"].device)

        opacity_sparse_weight = self.opacity_sparse_weight
        in_shape_weight = self.in_shape_weight
        if self.dynamic_weight:
            opacity_sparse_weight *= 1 + curr_epoch_for_loss**2 / self.opacity_sparse_denom
            in_shape_weight *= max(0.0, (1 - curr_epoch_for_loss / self.milestone))

        loss = (
            rgb_loss
            + self.eikonal_weight * eikonal_loss
            + self.bce_weight * bce_loss
            + opacity_sparse_weight * opacity_sparse_loss
            + in_shape_weight * in_shape_loss
            + self.sdf_loss_weight * sdf_loss
        )

        return {
            "loss": loss,
            "rgb_loss": rgb_loss,
            "eikonal_loss": eikonal_loss,
            "bce_loss": bce_loss,
            "opacity_sparse_loss": opacity_sparse_loss,
            "in_shape_loss": in_shape_loss,
            "sdf_loss": sdf_loss,
        }
