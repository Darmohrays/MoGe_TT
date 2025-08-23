from typing import Tuple, Union, Optional, Dict

import torch
import kornia.augmentation as K
from kornia.augmentation import AugmentationSequential


# Configure for GPU usage - more comprehensive setup
from .render import render_pcd_to_numpy_open3d
import open3d as o3d


def _rodrigues_pt(omega: torch.Tensor) -> torch.Tensor:
    """
    PyTorch implementation of the Rodrigues' rotation formula.
    Converts an axis-angle vector to a 3x3 rotation matrix.

    Args:
        omega (torch.Tensor): A (3,) tensor representing the axis-angle vector in radians.

    Returns:
        torch.Tensor: The corresponding (3, 3) rotation matrix.
    """
    theta = torch.linalg.norm(omega)
    if theta < 1e-12:
        return torch.eye(3, dtype=omega.dtype, device=omega.device)

    k = omega / theta
    # Create the skew-symmetric cross-product matrix K
    K = torch.zeros((3, 3), dtype=omega.dtype, device=omega.device)
    K[0, 1], K[0, 2] = -k[2], k[1]
    K[1, 0], K[1, 2] = k[2], -k[0]
    K[2, 0], K[2, 1] = k[1], -k[0]

    s, c = torch.sin(theta), torch.cos(theta)
    I = torch.eye(3, dtype=omega.dtype, device=omega.device)
    
    # Rodrigues' formula: R = I + sin(theta)*K + (1-cos(theta))*K^2
    return I + s * K + (1 - c) * torch.matmul(K, K)

def _euler_xyz_to_R_pt(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """
    PyTorch implementation to convert Euler angles to a rotation matrix.
    The rotation order is R = Rz @ Ry @ Rx.

    Args:
        roll (torch.Tensor): Rotation about the x-axis in radians (scalar tensor).
        pitch (torch.Tensor): Rotation about the y-axis in radians (scalar tensor).
        yaw (torch.Tensor): Rotation about the z-axis in radians (scalar tensor).

    Returns:
        torch.Tensor: The corresponding (3, 3) rotation matrix.
    """
    device, dtype = roll.device, roll.dtype
    
    cx, sx = torch.cos(roll), torch.sin(roll)
    cy, sy = torch.cos(pitch), torch.sin(pitch)
    cz, sz = torch.cos(yaw), torch.sin(yaw)

    Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=dtype, device=device)
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=dtype, device=device)
    Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=dtype, device=device)

    return torch.matmul(Rz, torch.matmul(Ry, Rx))

def _sample_translation_pt(
    trans_range: Union[float, tuple, list],
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    PyTorch implementation to sample a 3D translation vector.

    Args:
        trans_range:
          - scalar r  -> each axis ~ U(-r, r)
          - (min, max)-> each axis ~ U(min, max)
          - ((xmin,xmax),(ymin,ymax),(zmin,zmax)) per-axis ranges
        generator (torch.Generator, optional): PyTorch random number generator for reproducibility.
        device (torch.device, optional): The device to create the tensor on.
        dtype (torch.dtype): The data type of the output tensor.

    Returns:
        torch.Tensor: A (3,) translation vector.
    """
    # Helper to generate uniform random numbers in a range
    def rand_unif(low, high, size, gen):
        return (high - low) * torch.rand(*size, generator=gen, device=device, dtype=dtype) + low

    if isinstance(trans_range, (int, float)):
        r = float(abs(trans_range))
        return rand_unif(-r, r, (3,), generator)
    
    arr = torch.tensor(trans_range, dtype=dtype, device=device)
    if arr.shape == (2,):
        lo, hi = arr
        return rand_unif(lo, hi, (3,), generator)
    
    if arr.shape == (3, 2):
        return torch.tensor([
            rand_unif(lo, hi, (1,), generator).item() for lo, hi in arr
        ], dtype=dtype, device=device)
        
    raise ValueError("trans_range must be scalar, (min,max), or ((xmin,xmax),(ymin,ymax),(zmin,zmax)).")


def jitter_pcd_pt(
    pcd: torch.Tensor,
    *,
    mode: str = "euler",
    roll_deg: Tuple[float, float] = (-1.0, 1.0),
    pitch_deg: Tuple[float, float] = (-1.0, 1.0),
    yaw_deg: Tuple[float, float] = (-1.0, 1.0),
    sigma_deg: float = 3.0,
    trans_range: Union[float, tuple, list] = 0.02,
    generator: Optional[torch.Generator] = None,
    inplace: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    ... (docstring remains the same) ...
    """
    device, dtype = pcd.device, pcd.dtype
    
    # MODIFIED: Use the provided generator or create a new one if None
    if generator is None:
        generator = torch.Generator(device=device)

    # --- Rotation
    if mode.lower() == "euler":
        r_rad = (roll_deg[1] - roll_deg[0]) * torch.rand(1, generator=generator, device=device, dtype=dtype) + roll_deg[0]
        p_rad = (pitch_deg[1] - pitch_deg[0]) * torch.rand(1, generator=generator, device=device, dtype=dtype) + pitch_deg[0]
        y_rad = (yaw_deg[1] - yaw_deg[0]) * torch.rand(1, generator=generator, device=device, dtype=dtype) + yaw_deg[0]
        
        # Convert to radians for the rotation matrix function
        r, p, y = r_rad * (torch.pi / 180.0), p_rad * (torch.pi / 180.0), y_rad * (torch.pi / 180.0)
        
        R = _euler_xyz_to_R_pt(r, p, y)
        angles_deg = (r_rad.item(), p_rad.item(), y_rad.item())

    elif mode.lower() == "axis_angle":
        # Isotropic small rotation: random axis, small angle ~ N(0, sigma)
        axis = torch.randn(3, generator=generator, device=device, dtype=dtype)
        axis /= (torch.linalg.norm(axis) + 1e-12)
        
        angle_deg = torch.randn(1, generator=generator, device=device, dtype=dtype) * sigma_deg
        angle_rad = angle_deg * (torch.pi / 180.0)
        
        R = _rodrigues_pt(axis * angle_rad)
        angles_deg = None  # Not Euler-based

    else:
        raise ValueError("mode must be 'euler' or 'axis_angle'.")

    # --- Translation
    t = _sample_translation_pt(trans_range, generator=generator, device=device, dtype=dtype)

    # --- 4x4 transform matrix
    T = torch.eye(4, dtype=dtype, device=device)
    T[:3, :3] = R
    T[:3, 3] = t

    # --- Apply to point cloud
    # Transform: pcd_out = (R @ pcd.T).T + t
    pcd_transformed = torch.matmul(pcd, R.T) + t

    if inplace:
        pcd.copy_(pcd_transformed)
        pcd_out = pcd
    else:
        pcd_out = pcd_transformed

    info = {"angles_deg_xyz": angles_deg, "translation_m_xyz": t.cpu().numpy()}
    return pcd_out, T, info


def _geom_params_from_severity(sev: int, mode: str) -> dict:
    """Return default jitter params for given severity and mode."""
    # Max Euler yaw/pitch/roll magnitude (degrees)
    rot_max_deg = [0.0, 0.5, 1.0, 3.0, 7.5, 15.0][sev]
    
    # Translation range for x, y axes (pan/tilt)
    trans_max_xy = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10][sev]
    
    # Translation range for z-axis (zoom), set to be larger than x/y
    trans_max_z = [0.0, 0.01, 0.025, 0.05, 0.1, 0.20][sev]

    # Construct the per-axis translation range tuple for _sample_translation_pt
    trans_range = (
        (-trans_max_xy, trans_max_xy),  # (xmin, xmax)
        (-trans_max_xy, trans_max_xy),  # (ymin, ymax)
        (-trans_max_z*50, 0)     # (zmin, zmax)
    )

    params = {"trans_range": trans_range}
    if mode == "euler":
        r = (-rot_max_deg, rot_max_deg)
        params.update(dict(roll_deg=r, pitch_deg=r, yaw_deg=r))
    else:  # axis_angle
        params.update(dict(sigma_deg=rot_max_deg))
        
    return params
    

def _build_color_augmentation(sev: int):
        if sev == 0:
            return None  # handled downstream (no-op)

        # Ranges increase with severity
        # Shared ±range for brightness/contrast/saturation
        bcs = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50][sev]
        hue_r = [0.0, 0.03, 0.06, 0.10, 0.20, 0.30][sev]
        gamma_amp = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50][sev]

        # Probabilities scale with severity
        p_main = min(0.16 * sev, 0.95)              # for ColorJitter
        p_aux  = min(0.12 * sev, 0.8)               # for hue/sat/auto-contrast
        p_gs   = min(0.06 * sev, 0.3)               # grayscale
        p_gamma= min(0.10 * sev, 0.7)               # gamma

        gamma_lo = max(1.0 - gamma_amp, 0.5)
        gamma_hi = 1.0 + gamma_amp
        gain_lo  = max(1.0 - 0.1 * sev, 0.5)
        gain_hi  = 1.0 + 0.1 * sev

        # Compose Kornia augmentations
        transforms = [
            K.ColorJitter(
                brightness=bcs, contrast=bcs, saturation=bcs, hue=hue_r, p=p_main
            ),
            K.RandomGamma(gamma=(gamma_lo, gamma_hi), gain=(gain_lo, gain_hi), p=p_gamma),
            K.RandomHue(hue=(-hue_r, hue_r), p=p_aux),
            K.RandomSaturation(saturation=(max(1 - bcs, 0.1), 1 + bcs), p=p_aux),
            K.RandomGrayscale(p=p_gs),
            K.RandomAutoContrast(p=p_aux),
        ]

        # Apply up to N transforms per call (bounded by list length)
        random_apply = max(1, min(1 + 2 * sev, len(transforms)))

        return AugmentationSequential(
            *transforms,
            random_apply=random_apply,
            data_keys=["input"],
        )


def generate_jittered_batch(
    image: torch.Tensor,
    moge_output: Dict[str, torch.Tensor],
    batch_size: int,
    random_state: Optional[int] = None,
    geometric_aug_severity: int = 3,
    color_aug_severity: int = 3,
    **jitter_kwargs
) -> dict:
    """
    Generates a batch of jittered point clouds and rendered images from a single source.
    """
    # ---- Validate severities
    if not (isinstance(geometric_aug_severity, int) and 0 <= geometric_aug_severity <= 5):
        raise ValueError("geometric_aug_severity must be an int in [0, 5].")
    if not (isinstance(color_aug_severity, int) and 0 <= color_aug_severity <= 5):
        raise ValueError("color_aug_severity must be an int in [0, 5].")

    point_map = moge_output['points']
    mask = moge_output['mask']

    height, width = image.shape[1:]
    fx = moge_output['intrinsics'][0, 0].item() * width
    fy = moge_output['intrinsics'][1, 1].item() * height
    cx = moge_output['intrinsics'][0, 2].item() * width
    cy = moge_output['intrinsics'][1, 2].item() * height

    pcd = point_map

    # Optional master RNG for reproducibility
    gen_master = None
    if random_state is not None:
        gen_master = torch.Generator(device=pcd.device)
        gen_master.manual_seed(int(random_state))

    # Pre-build color augmentation pipeline
    color_augmentation = _build_color_augmentation(color_aug_severity)

    pcds_list = []
    transforms_list = []
    transforms_inv_list = []
    images = []
    masks = []
    intrinsics = []

    # Determine jitter mode (default 'euler'), then severity->params
    mode = jitter_kwargs.get("mode", "euler").lower()
    force_no_geom = (geometric_aug_severity == 0)

    for i in range(batch_size):
        intrinsics.append(moge_output['intrinsics'].detach().clone())

        # Per-item RNG (uses master if provided)
        child_seed = torch.randint(
            0, 2**63 - 1, (1,),
            device=pcd.device,
            generator=gen_master
        ).item()
        gen_i = torch.Generator(device=pcd.device)
        gen_i.manual_seed(child_seed)

        # Severity-derived defaults; user jitter_kwargs can still override
        base_geom = _geom_params_from_severity(geometric_aug_severity, mode)
        if force_no_geom:
            # Hard override to guarantee NO geometric changes
            geom_kwargs = {**jitter_kwargs}
            geom_kwargs.update(dict(trans_range=0.0))
            if mode == "euler":
                geom_kwargs.update(dict(roll_deg=(0.0, 0.0),
                                        pitch_deg=(0.0, 0.0),
                                        yaw_deg=(0.0, 0.0)))
            else:
                geom_kwargs.update(dict(sigma_deg=0.0))
        else:
            # Defaults from severity, user-provided values take precedence
            geom_kwargs = {**base_geom, **jitter_kwargs}

        pcd_flatten = pcd.reshape(-1, 3)
        pcd_out, T, _ = jitter_pcd_pt(
            pcd_flatten,
            generator=gen_i,
            **geom_kwargs
        )

        # Inverse transform
        R = T[:3, :3]
        t = T[:3, 3]
        R_T = R.T
        t_inv = -torch.matmul(R_T, t)

        T_inv = torch.eye(4, dtype=T.dtype, device=T.device)
        T_inv[:3, :3] = R_T
        T_inv[:3, 3] = t_inv

        assert torch.equal(pcd_out, torch.matmul(pcd_flatten, R.T) + t)
        assert torch.allclose(pcd_flatten, torch.matmul(pcd_out - t, R), atol=0.00005)

        pcd_out = pcd_out.reshape(pcd.shape)

        # pcds_list.append(pcd_out)
        transforms_list.append(T)
        transforms_inv_list.append(T_inv)

        # ---- Color augmentation
        if color_augmentation is None:
            image_aug = image  # no-op at severity 0
        else:
            image_aug = color_augmentation(image)[0]

        # Open3D render
        if geometric_aug_severity == 0:
            rendered_image = image_aug
            bg_mask = mask.detach().clone()
        else:
            # Colors for visible (masked) points
            pcd_out_i = pcd_out
            pcd_out_masked = pcd_out_i[mask]
            colors_masked = image_aug[:, mask].T
            pcd_open3d = o3d.geometry.PointCloud()
            pcd_open3d.points = o3d.utility.Vector3dVector(pcd_out_masked.cpu().numpy())
            pcd_open3d.colors = o3d.utility.Vector3dVector(colors_masked.cpu().numpy())

            rendered_image, rendered_pcd, bg_mask = render_pcd_to_numpy_open3d(
                pcd_open3d,
                width=width, height=height,
                fx=fx, fy=fy, cx=cx, cy=cy,
                bg_color=(255, 255, 255)
            )
            fg_mask = ~bg_mask
            rendered_image = torch.from_numpy(rendered_image).float().permute(2, 0, 1) / 255.0
        
        pcds_list.append(torch.from_numpy(rendered_pcd))
        images.append(rendered_image)
        masks.append(torch.tensor(fg_mask))

        # Optional debug dumps
        import cv2
        import numpy as np
        np.save(f"{i}_pcd.npy", rendered_pcd)
        np.save(f"{i}_mask.npy", fg_mask)
        image_to_save = (rendered_image.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        cv2.imwrite(f"{i}.png", cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR))
        
        # import ipdb; ipdb.set_trace()

    return {
        "images": torch.stack(images).to(pcd.device),
        "masks_gt": torch.stack(masks).to(pcd.device),
        "points_gt": torch.stack(pcds_list).to(pcd.device),
        "transforms": torch.stack(transforms_list).to(pcd.device),
        "transforms_inv": torch.stack(transforms_inv_list).to(pcd.device),
        "intrinsics": torch.stack(intrinsics).to(pcd.device)
    }