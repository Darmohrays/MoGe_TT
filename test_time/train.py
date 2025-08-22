import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

# Assuming the following utility functions and modules are available from your project
# If they are in different locations, you might need to adjust the imports.
from moge.train.losses import (
    affine_invariant_global_loss,
    affine_invariant_local_loss,
    edge_loss,
    normal_loss,
    mask_l2_loss,
    mask_bce_loss,
    monitoring,
)
from moge.utils.tools import flatten_nested_dict, key_average
import utils3d # Assuming this is a custom module for 3D utilities

# You can place this function in a separate utils file or directly in your training script.
def calculate_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    device: torch.device
) -> Dict[str, float]:
    """
    Calculates the loss for a given model and a single batch of data.

    Args:
        model (nn.Module): The neural network model to evaluate.
        batch (Dict[str, torch.Tensor]): A dictionary containing the batch data
                                         (images, ground truth depth, masks, etc.).
        config (Dict[str, Any]): A configuration dictionary, expected to have a 'loss'
                                 key specifying loss functions and weights.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') to perform
                               computations on.

    Returns:
        Dict[str, float]: A dictionary containing the total loss and the value of
                          each individual loss component, averaged over the batch.
    """
    # 1. Unpack batch and move data to the specified device
    image = batch['image'].to(device)
    gt_depth = batch['depth'].to(device)
    gt_mask = batch['depth_mask'].to(device)
    gt_mask_fin = batch['depth_mask_fin'].to(device)
    gt_mask_inf = batch['depth_mask_inf'].to(device)
    gt_intrinsics = batch['intrinsics'].to(device)
    label_type = batch['label_type']
    current_batch_size = image.shape[0]

    # Skip batch if all labels are invalid to prevent errors
    if all(label == 'invalid' for label in label_type):
        return {'total_loss': 0.0}

    # 2. Prepare ground truth data
    gt_points = utils3d.torch.depth_to_points(gt_depth, intrinsics=gt_intrinsics)
    gt_focal = 1 / (1 / gt_intrinsics[..., 0, 0] ** 2 + 1 / gt_intrinsics[..., 1, 1] ** 2) ** 0.5

    # 3. Forward pass
    # Using a fixed number of tokens for simplicity, as it was randomized in the original loop
    num_tokens = config['model']['num_tokens_range'][1]
    output = model(image, num_tokens=num_tokens)
    pred_points, pred_mask = output['points'], output['mask']

    # 4. Compute loss for each instance in the batch
    batch_records = []
    total_loss_list = []

    for i in range(current_batch_size):
        loss_dict, weight_dict, misc_dict = {}, {}, {}
        gt_metric_scale = None # Used for affine invariant losses

        # Add monitoring metrics
        misc_dict['monitoring'] = monitoring(pred_points[i])

        # Get the loss configuration for the current instance's label type
        loss_config = config['loss'][label_type[i]]

        # Calculate each specified loss component
        for name, params in loss_config.items():
            weight_dict[name] = params['weight']
            loss_func_name = params['function']

            if loss_func_name == 'affine_invariant_global_loss':
                loss_dict[name], misc_dict[name], gt_metric_scale = affine_invariant_global_loss(
                    pred_points[i], gt_points[i], gt_mask[i], **params['params']
                )
            elif loss_func_name == 'affine_invariant_local_loss':
                loss_dict[name], misc_dict[name] = affine_invariant_local_loss(
                    pred_points[i], gt_points[i], gt_mask[i], gt_focal[i], gt_metric_scale, **params['params']
                )
            elif loss_func_name == 'normal_loss':
                loss_dict[name], misc_dict[name] = normal_loss(
                    pred_points[i], gt_points[i], gt_mask[i]
                )
            elif loss_func_name == 'edge_loss':
                loss_dict[name], misc_dict[name] = edge_loss(
                    pred_points[i], gt_points[i], gt_mask[i]
                )
            elif loss_func_name == 'mask_bce_loss':
                loss_dict[name], misc_dict[name] = mask_bce_loss(
                    pred_mask[i], gt_mask_fin[i], gt_mask_inf[i]
                )
            elif loss_func_name == 'mask_l2_loss':
                loss_dict[name], misc_dict[name] = mask_l2_loss(
                    pred_mask[i], gt_mask_fin[i], gt_mask_inf[i]
                )
            else:
                raise ValueError(f"Undefined loss function: {loss_func_name}")

        # Flatten nested dictionaries for easier processing
        flat_weights = {'.'.join(k): v for k, v in flatten_nested_dict(weight_dict).items()}
        flat_losses = {'.'.join(k): v for k, v in flatten_nested_dict(loss_dict).items()}
        flat_misc = {'.'.join(k): v for k, v in flatten_nested_dict(misc_dict).items()}

        # Calculate the weighted sum of losses for the current instance
        instance_total_loss = sum(
            flat_weights[k] * flat_losses[k] for k in flat_losses
        )
        total_loss_list.append(instance_total_loss)

        # Store results for averaging later
        record = {k: v.item() for k, v in flat_losses.items()}
        record.update(flat_misc)
        batch_records.append(record)

    # 5. Aggregate losses for the entire batch
    # Average the total loss across all instances
    final_total_loss = sum(total_loss_list) / len(total_loss_list) if total_loss_list else torch.tensor(0.0)

    # Average each individual loss component and metric
    averaged_records = key_average(batch_records)
    averaged_records['total_loss'] = final_total_loss.item()

    return averaged_records