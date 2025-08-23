import torch
import torch.nn.functional as F
import utils3d

from moge.train.losses import (
    affine_invariant_global_loss,
    affine_invariant_local_loss, 
    edge_loss,
    normal_loss, 
    mask_l2_loss, 
    mask_bce_loss,
    monitoring, 
)
from moge.utils.tools import flatten_nested_dict

def _apply_transform(points_bhw3: torch.Tensor, T_b44: torch.Tensor) -> torch.Tensor:
    """
    Apply a 4x4 transform to a dense point-map.

    Args:
        points_bhw3: (B,H,W,3) points in the *source* camera frame.
        T_b44:      (B,4,4) transform that maps source->target camera frame.
                    We assume column-vector convention: X' = T @ [X;1].
                    With row vectors, this code uses T^T (see bmm below).

    Returns:
        points'_bhw3: (B,H,W,3) points in the *target* camera frame.
    """
    B, H, W, _ = points_bhw3.shape
    ones = torch.ones((B, H, W, 1), device=points_bhw3.device, dtype=points_bhw3.dtype)
    P_h = torch.cat([points_bhw3, ones], dim=-1).view(B, -1, 4)                 # (B, H*W, 4)
    # Row-vector multiply by T^T so it's equivalent to column-vector T @ [x,y,z,1]^T
    Pp_h = torch.bmm(P_h, T_b44.transpose(1, 2))                                # (B, H*W, 4)
    Pp = Pp_h[..., :3].view(B, H, W, 3)
    return Pp

def _grayscale(images_bchw: torch.Tensor) -> torch.Tensor:
    """Convert Bx3xHxW to Bx1xHxW luminance (no grad-breaking ops)."""
    r, g, b = images_bchw[:, 0:1], images_bchw[:, 1:2], images_bchw[:, 2:3]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

def _edge_aware_smoothness(depth_bhw: torch.Tensor,
                           image_bchw: torch.Tensor,
                           valid_bhw: torch.Tensor,
                           gamma: float = 10.0,
                           eps: float = 1e-6) -> torch.Tensor:
    """
    Edge-aware first-order smoothness on depth, weighted by image gradients.
    depth_bhw:  (B,H,W)
    image_bchw: (B,3,H,W)
    valid_bhw:  (B,H,W) boolean/float mask for pixels with reliable geometry
    """
    B, H, W = depth_bhw.shape
    gray = _grayscale(image_bchw)

    # Finite differences (forward) for depth and image
    dz_dx = torch.abs(depth_bhw[:, :, 1:] - depth_bhw[:, :, :-1])               # (B,H,W-1)
    dz_dy = torch.abs(depth_bhw[:, 1:, :] - depth_bhw[:, :-1, :])               # (B,H-1,W)

    Ix = torch.mean(torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1]), dim=1)   # (B,H,W-1)
    Iy = torch.mean(torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :]), dim=1)   # (B,H-1,W)

    w_x = torch.exp(-gamma * Ix)                                                # edge-aware weights
    w_y = torch.exp(-gamma * Iy)

    # Valid masks for gradient pairs (both pixels must be valid)
    vx = (valid_bhw[:, :, 1:] > 0.5) & (valid_bhw[:, :, :-1] > 0.5)
    vy = (valid_bhw[:, 1:, :] > 0.5) & (valid_bhw[:, :-1, :] > 0.5)

    loss_x = (dz_dx * w_x * vx.float()).sum()
    loss_y = (dz_dy * w_y * vy.float()).sum()

    denom = vx.float().sum() + vy.float().sum() + eps
    return (loss_x + loss_y) / denom

def _normals_from_points(points_bhw3: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Estimate per-pixel normals from a dense point map using forward differences and
    a cross product. Output normals are L2-normalized and defined on the interior
    (H-1, W-1); we pad to (H, W) by replicating border values.
    """
    B, H, W, _ = points_bhw3.shape
    du = points_bhw3[:, :, 1:, :] - points_bhw3[:, :, :-1, :]      # (B,H,W-1,3) width diff
    dv = points_bhw3[:, 1:, :, :] - points_bhw3[:, :-1, :, :]      # (B,1? no) (B,H-1,W,3) height diff

    # Align shapes to the common (H-1, W-1)
    du_c = du[:, :-1, :, :]                                        # (B,H-1,W-1,3)
    dv_c = dv[:, :, :-1, :]                                        # (B,H-1,W-1,3)

    n = torch.linalg.cross(dv_c, du_c, dim=-1)                     # (B,H-1,W-1,3)
    n = F.normalize(n, dim=-1, eps=eps)

    # Pad back to (H, W) by replicating last row/col (no gradients lost)
    n = F.pad(n.permute(0, 3, 1, 2), (0, 1, 0, 1), mode="replicate")  # (B,3,H,W)
    return n.permute(0, 2, 3, 1).contiguous()                       # (B,H,W,3)

def compute_moge2_ttt_loss(
    batch: dict,
    *,
    use_transforms_inv: bool = False,
    # weights
    w_geom: float = 0.1,
    w_smooth: float = 0.05,
    w_normal: float = 0.1,
    # w_geom: float = 1.0,
    # w_smooth: float = 0.0,
    # w_normal: float = 0.,
    gamma_edge: float = 10.0,
    eps: float = 1e-6,
    **kwargs
):
    """
    Compute TTT losses given:
      images        : (B,3,H,W)   current (jittered) views used for student prediction
      masks_gt      : (B,H,W)     boolean visibility/occlusion mask from teacher render
      points_gt     : (B,H*W,3)   teacher point map in the *source* camera frame
      transforms    : (B,4,4)     transform mapping source -> target camera frame
      transforms_inv: (B,4,4)     inverse (target -> source), if needed
      points        : (B,H,W,3)   student-predicted point map in the *target* camera frame
      mask          : (B,H,W)     student reliability mask/probability (0..1)

    Assumptions:
      - `transforms` maps the teacher's source camera to the student's target camera.
        If your tensors are the other way around, set `use_transforms_inv=True`.
      - All point coordinates are in meters in their respective camera frames.
    Returns:
      dict with 'loss', and individual components.
    """
    images: torch.Tensor        = batch["images"]
    masks_gt: torch.Tensor      = batch["masks_gt"]          # bool
    points_gt: torch.Tensor = batch["points_gt"]         # (B,H*W,3)
    T: torch.Tensor             = batch["transforms_inv"] if use_transforms_inv else batch["transforms"]
    points_student: torch.Tensor= batch["points"]            # (B,H,W,3)
    mask_student: torch.Tensor  = batch["mask"]              # (B,H,W)

    # R = T[0, :3, :3]
    # t = T[0, :3, 3]

    test_idx = 0
    B, C, H, W = images.shape

    # Transform teacher (source) points into the *target* camera frame
    # points_gt_in_target = _apply_transform(points_gt, T)     # (B,H,W,3)
    points_gt_in_target = points_gt

    # Validity mask: teacher-visible AND student-valid AND finite AND positive depth
    finite_gt   = torch.isfinite(points_gt_in_target).all(dim=-1)
    finite_pred = torch.isfinite(points_student).all(dim=-1)
    posz_gt     = points_gt_in_target[..., 2] > eps
    posz_pred   = points_student[..., 2] > eps

    valid = (masks_gt.bool() & finite_gt & finite_pred & posz_gt & posz_pred)
    if mask_student is not None:
        valid = valid & (mask_student > 0.5)

    valid_f = valid.float()
    num_valid = valid_f.sum().clamp_min(1.0)

    # ---------------------------
    # 1) Geometry-equivariance (L1 in meters)
    # ---------------------------
    geom_residual = torch.abs(points_student - points_gt_in_target).sum(dim=-1)   # (B,H,W)
    geom_loss = (geom_residual * valid_f).sum() / num_valid

    # import numpy as np
    # np.save("points_student.npy", points_student[test_idx].detach().cpu().numpy())
    # np.save("points_gt_in_target.npy", points_gt_in_target[test_idx].detach().cpu().numpy())
    # np.save("valid.npy", valid[test_idx].detach().cpu().numpy())
    # np.save("image.npy", images[0].permute(1, 2, 0).detach().cpu().numpy())

    # import ipdb; ipdb.set_trace()
    # ---------------------------
    # 2) Edge-aware smoothness on student depth
    # ---------------------------
    depth_student = points_student[..., 2]                                         # (B,H,W)
    smooth_loss = _edge_aware_smoothness(depth_student, images, valid_f, gamma=gamma_edge, eps=eps)

    # ---------------------------
    # 3) Normal consistency (student vs teacher-in-target)
    # ---------------------------
    n_student = _normals_from_points(points_student, eps=eps)                      # (B,H,W,3)
    n_gt      = _normals_from_points(points_gt_in_target.detach(), eps=eps)        # (B,H,W,3)  (stop-grad teacher)

    # To avoid border effects, drop last row/col (their normals came from replicated pads)
    n_valid = valid[:, :-1, :-1]
    n_diff = torch.abs(n_student[:, :-1, :-1, :] - n_gt[:, :-1, :-1, :]).sum(dim=-1)  # (B,H-1,W-1)
    normal_loss = (n_diff * n_valid.float()).sum() / (n_valid.float().sum().clamp_min(1.0))

    # ---------------------------
    # Total
    # ---------------------------
    loss = w_geom * geom_loss + w_smooth * smooth_loss + w_normal * normal_loss

    return {
        "loss": loss,
        "geom_loss": geom_loss.detach(),
        "smooth_loss": smooth_loss.detach(),
        "normal_loss": normal_loss.detach(),
        "valid_fraction": (num_valid / (B * H * W)).detach(),
    }

def compute_moge2_ttt_loss_from_orig(batch: dict, config: dict, device: str,
                                     use_transforms_inv: bool):
    loss_dict, weight_dict, misc_dict = {}, {}, {}
    image, gt_points, gt_mask, gt_intrinsics = batch['images'], batch['points_gt'], batch['masks_gt'], batch['intrinsics']
    image, gt_points, gt_mask, gt_intrinsics = image.to(device), gt_points.to(device), gt_mask.to(device), gt_intrinsics.to(device)

    T: torch.Tensor = batch["transforms_inv"] if use_transforms_inv else batch["transforms"]

    current_batch_size = image.shape[0]

    pred_points, pred_mask, pred_metric_scale = batch['points'], batch['mask'], batch.get('metric_scale', None)

    # gt_points = utils3d.torch.depth_to_points(gt_depth, intrinsics=gt_intrinsics)
    # gt_points = _apply_transform(gt_points, T)
    gt_focal = 1 / (1 / gt_intrinsics[..., 0, 0] ** 2 + 1 / gt_intrinsics[..., 1, 1] ** 2) ** 0.5
    
    loss_list = []
    for i in range(current_batch_size):
        for k, v in config['loss'].items():
            weight_dict[k] = v['weight']
            if v['function'] == 'affine_invariant_global_loss':
                loss_dict[k], misc_dict[k], gt_metric_scale = affine_invariant_global_loss(pred_points[i], gt_points[i], gt_mask[i], **v['params'])
            elif v['function'] == 'affine_invariant_local_loss':
                loss_dict[k], misc_dict[k] = affine_invariant_local_loss(pred_points[i], gt_points[i], gt_mask[i], gt_focal[i], gt_metric_scale, **v['params'])
            elif v['function'] == 'normal_loss':
                loss_dict[k], misc_dict[k] = normal_loss(pred_points[i], gt_points[i], gt_mask[i])
            elif v['function'] == 'edge_loss':
                loss_dict[k], misc_dict[k] = edge_loss(pred_points[i], gt_points[i], gt_mask[i])
            elif v['function'] == 'mask_bce_loss':
                loss_dict[k], misc_dict[k] = mask_bce_loss(pred_mask[i], gt_mask[i], gt_mask[i])
            elif v['function'] == 'mask_l2_loss':
                loss_dict[k], misc_dict[k] = mask_l2_loss(pred_mask[i], gt_mask[i], gt_mask[i])
            else:
                raise ValueError(f'Undefined loss function: {v["function"]}')
            
        weight_dict = {'.'.join(k): v for k, v in flatten_nested_dict(weight_dict).items()}
        loss_dict = {'.'.join(k): v for k, v in flatten_nested_dict(loss_dict).items()}
        loss_ = sum([weight_dict[k] * loss_dict[k] for k in loss_dict], start=torch.tensor(0.0, device=device))
        loss_list.append(loss_)

        return dict(loss=sum(loss_list) / len(loss_list))