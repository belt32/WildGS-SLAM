from typing import Dict, Tuple
import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from thirdparty.gaussian_splatting.utils.loss_utils import ssim
from src.utils.dyn_uncertainty import mapping_utils as map_utils
import cv2
import numpy as np
from src.utils.thermal_prior_utils import get_calibrated_thermal_mask


def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def get_loss_tracking(config, image, depth, opacity, viewpoint, monocular=True, uncertainty=None):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if monocular:
        return get_loss_tracking_rgb(config, image_ab, opacity, viewpoint, uncertainty)
    else:
        raise NotImplementedError(f"Only implemented monocular, not rgbd for uncertainty-aware tracking")
        # return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint, uncertainty)

def get_loss_tracking_rgb(config, image, opacity, viewpoint, uncertainty=None):
    """Compute RGB tracking loss between rendered and ground truth images.
    This function adds uncertainty mask on the original function from MonoGS
    
    Args:
        config: Configuration dictionary containing training parameters
        image: Rendered RGB image tensor (3, H, W)
        opacity: Opacity tensor (1, H, W) 
        viewpoint: Camera object containing ground truth image and gradient mask
        uncertainty: Optional uncertainty estimates (H, W)
    
    Returns:
        Scalar loss tensor
    """
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)

    # Create mask for valid RGB pixels above intensity threshold
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask

    # Compute L1 loss weighted by opacity
    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    if uncertainty is not None:
        # Weight loss inversely proportional to uncertainty
        # Higher uncertainty -> lower weight 
        # Zero out weights below 0.1 to ignore highly uncertain regions (Todo: verify this is useful)
        weights = 0.5 / (uncertainty.unsqueeze(0))**2
        weights = torch.where(weights < 0.1, 0.0, weights)
        l1 *= weights
    return l1.mean()

# Not used, but kept for reference
def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, uncertainty=None
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()


def get_loss_mapping(config, image, depth, viewpoint, initialization=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b

    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)


def get_loss_mapping_rgbd(config, image, depth, viewpoint):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[
        None
    ]
    loss = 0
    if config["Training"]["ssim_loss"]:
        ssim_loss = 1.0 - ssim(image, gt_image)

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    if config["Training"]["ssim_loss"]:
        hyperparameter = config["opt_params"]["lambda_dssim"]
        loss += (1.0 - hyperparameter) * l1_rgb + hyperparameter * ssim_loss
    else:
        loss += l1_rgb

    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

    return alpha * loss.mean() + (1 - alpha) * l1_depth.mean()


def get_loss_mapping_uncertainty(
    rendered_img, rendered_depth, opacity, viewpoint, uncertainty_network, map_utils,
    config, train_frac=0.5, ssim_frac=0.5, initialization=False, freeze_uncertainty_loss=False,
    thermal_mask=None, thermal_image=None, T_rgb_thermal=None, K_thermal=None,
    gaussians=None, thermal_info=None
):
    """
    Compute the mapping and uncertainty loss for SLAM.

    Args:
        rendered_img: Rendered RGB image (torch.Tensor).
        rendered_depth: Rendered depth map (torch.Tensor).
        opacity: Opacity map (torch.Tensor).
        viewpoint: Viewpoint object containing ground truth data and features.
        uncertainty_network: Network to predict uncertainty from features.
        map_utils: Utility functions for mapping loss computation.
        config: Configuration dictionary.
        train_frac: Fraction of training data to use for loss computation.
        ssim_frac: Fraction of SSIM loss to use in RGB loss.
        initialization: Whether the SLAM system is in initialization phase.
        freeze_uncertainty_loss: Whether to freeze the uncertainty loss.
        thermal_mask: Thermal mask for penalizing certain regions (torch.Tensor, optional).
        thermal_image: Thermal image data (torch.Tensor, optional).
        T_rgb_thermal: Transformation matrix from RGB to thermal camera (torch.Tensor, optional).
        K_thermal: Intrinsic matrix of the thermal camera (torch.Tensor, optional).
        gaussians: Gaussian components for SLAM (optional).
        thermal_info: Dictionary containing thermal image data and calibration info (optional).

    Returns:
        uncertainty: Predicted uncertainty map (torch.Tensor).
        total_loss: Combined mapping and uncertainty loss (scalar).
    """
    # Apply exposure compensation if not initialization
    rendered_img = rendered_img if initialization else (
        torch.exp(viewpoint.exposure_a) * rendered_img + viewpoint.exposure_b
    )

    # Get config parameters
    alpha = config["Training"].get("alpha", 0.95)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    
    # Get reference data
    gt_img = viewpoint.original_image.cuda()
    ref_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=rendered_img.device
    )[None]

    # Create valid pixel mask
    _, h, w = gt_img.shape
    mask_shape = (1, h, w)
    rgb_pixel_mask = (gt_img.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)

    # Compute SSIM loss if enabled
    ssim_loss = 1.0 - ssim(rendered_img, gt_img) if config["Training"]["ssim_loss"] else 0.0

    # Predict uncertainty from features
    features = viewpoint.features.to(device=rendered_img.device)
    uncertainty = uncertainty_network(features)

     # Compute mapping losses with uncertainty
    uncer_loss, uncer_resized, l1_rgb, l1_depth = map_utils.compute_mapping_loss_components(
        gt_img,
        rendered_img,
        ref_depth,
        rendered_depth,
        uncertainty,
        opacity.view(*mask_shape),
        train_fraction=train_frac,
        ssim_fraction=ssim_frac,
        uncertainty_config=config["uncertainty_params"],
        mask=rgb_pixel_mask
    )

    # Combine RGB losses
    if config["Training"]["ssim_loss"]:
        lambda_dssim = config["opt_params"]["lambda_dssim"]
        rgb_loss = (1.0 - lambda_dssim) * l1_rgb + lambda_dssim * ssim_loss
    else:
        rgb_loss = l1_rgb

    # Apply uncertainty weighting
    weights = 0.5 / (uncer_resized.unsqueeze(0))**2
    # Zero out weights below 0.1 to ignore highly uncertain regions (Todo: verify this is useful)
    weights = torch.where(weights < 0.1, 0.0, weights)
    
    rgb_loss = weights * rgb_loss

    # Handle full resolution option
    if config['full_resolution']:
        weights = F.interpolate(
            weights.unsqueeze(0), 
            l1_depth.shape[-2:], 
            mode='bilinear'
        ).squeeze(0)

    # only add uncertainty on pixels where gt_depth < rendered_depth (add 1.0m buffer)
    # if you see a moving distractor, it must be closer to the camera than the static region
    # adding this can effectively remove some floater
    uncer_depth_mask = ref_depth < rendered_depth.detach() + 1.0
    l1_depth[uncer_depth_mask] = weights[uncer_depth_mask] * l1_depth[uncer_depth_mask]

    if freeze_uncertainty_loss:
        uncer_loss = uncer_loss.detach()

    # Combine all losses
    total_loss = (
        alpha * rgb_loss.mean() +
        (1 - alpha) * l1_depth.mean() +
        config["uncertainty_params"]["ssim_mult"] * uncer_loss.mean()
    )

    # Beta extension loss logic
    # a. Retrieve calibrated thermal mask
    thermal_mask = get_calibrated_thermal_mask(thermal_info['thermal_image'],
                                               thermal_info['T_rgb_thermal'],
                                               thermal_info['K_thermal'])

    # b. Create boundary region mask M_boundary
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=thermal_mask.device)
    dilated_mask = F.conv2d(thermal_mask[None, None].float(), kernel, padding=1).clamp(0, 1)
    eroded_mask = F.conv2d(thermal_mask[None, None].float(), -kernel, padding=1).clamp(0, 1)
    boundary_mask = (dilated_mask - eroded_mask).abs().squeeze(0).squeeze(0)

    # c. Compute beta extension loss
    beta_values = gaussians.get_beta_values_in_mask(boundary_mask)
    lambda_beta_ext = config['thermal'].get('lambda_beta_ext', 0.1)
    loss_beta_ext = lambda_beta_ext * beta_values.sum()

    # d. Add beta extension loss to total loss
    total_loss += loss_beta_ext

    # Log individual loss components
    rgb_loss_mean = rgb_loss.mean().item()
    depth_loss_mean = l1_depth.mean().item()
    uncer_loss_mean = uncer_loss.mean().item()
    beta_ext_loss = loss_beta_ext.item()

    print(f"[LOSSES] RGB: {rgb_loss_mean:.2f}, Depth: {depth_loss_mean:.2f}, Uncer: {uncer_loss_mean:.2f}, BetaExt: {beta_ext_loss:.2f}")

    return uncertainty, total_loss

def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()

def get_loss_mapping_rgbd(config, image, depth, viewpoint):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[
        None
    ]
    loss = 0
    if config["Training"]["ssim_loss"]:
        ssim_loss = 1.0 - ssim(image, gt_image)

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    if config["Training"]["ssim_loss"]:
        hyperparameter = config["opt_params"]["lambda_dssim"]
        loss += (1.0 - hyperparameter) * l1_rgb + hyperparameter * ssim_loss
    else:
        loss += l1_rgb

    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

    # Apply thermal mask-based penalties
    thermal_mask = None
    if thermal_mask is not None:
        thermal_mask = thermal_mask.to(device=rendered_img.device)
        thermal_penalty = config['Training'].get('thermal_penalty_weight', 1.0)

        # Scale RGB loss by thermal mask
        rgb_loss = rgb_loss + thermal_penalty * (1.0 - thermal_mask) * rgb_loss

        # Scale depth loss by thermal mask
        l1_depth = l1_depth + thermal_penalty * (1.0 - thermal_mask) * l1_depth

    return alpha * loss.mean() + (1 - alpha) * l1_depth.mean()