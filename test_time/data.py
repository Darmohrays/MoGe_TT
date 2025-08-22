from typing import Tuple, Union, Optional, Dict

import torch
import open3d as o3d
import kornia.augmentation as K
from kornia.augmentation import AugmentationSequential

from .render import render_pcd_to_numpy_open3d


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


def generate_jittered_batch(
    image: torch.Tensor,
    moge_output: Dict[str, torch.Tensor],
    batch_size: int,
    random_state: Optional[int] = None,
    **jitter_kwargs
) -> dict:
    """
    Generates a batch of jittered point clouds from a single source point cloud.

    Args:
        image (torch.Tensor): (3, W, H) normalized torch tensor
        pcd (torch.Tensor): The source (N, 3) point cloud.
        batch_size (int): The number of jittered versions to create.
        random_state (int): The master seed for the random number generator.
        **jitter_kwargs: Keyword arguments passed directly to jitter_pcd_pt.

    Returns:
        dict: A dictionary containing:
              - 'pcds': (B, N, 3) tensor of jittered point clouds.
              - 'transforms': (B, 4, 4) tensor of applied transforms.
              - 'transforms_inv': (B, 4, 4) tensor of inverse transforms.
    """
    point_map = moge_output['points']
    mask = moge_output['mask']

    height, width = image.shape[1:]
    fx = moge_output['intrinsics'][0, 0].item() * width
    fy = moge_output['intrinsics'][1, 1].item() * height
    cx = moge_output['intrinsics'][0, 2].item() * width
    cy = moge_output['intrinsics'][1, 2].item() * height

    pcd = point_map
    
    pcds_list = []
    transforms_list = []
    transforms_inv_list = []
    images = []
    masks = []
    intrinsics = []

    for i in range(batch_size):
        intrinsics.append(moge_output['intrinsics'].detach().clone())

        child_seed = torch.randint(0, 2**63 - 1, (1,),
                                   device=pcd.device).item()
        gen_i = torch.Generator(device=pcd.device)
        gen_i.manual_seed(child_seed)

        pcd_flatten = pcd.reshape(-1, 3)
        pcd_out, T, _ = jitter_pcd_pt(pcd_flatten,
                                      generator=gen_i,
                                      **jitter_kwargs)
        # Calculate the inverse transformation
        # T_inv = [ [R.T, -R.T @ t], [0, 1] ]
        R = T[:3, :3]
        t = T[:3, 3]
        R_T = R.T
        t_inv = -torch.matmul(R_T, t)
        
        T_inv = torch.eye(4, dtype=T.dtype, device=T.device)
        T_inv[:3, :3] = R_T
        T_inv[:3, 3] = t_inv

        # assert torch.equal(pcd_out, torch.matmul(pcd_flatten, R.T) + t)
        # assert torch.allclose(pcd_flatten, torch.matmul(pcd_out - t, R), rtol=0.00005)

        pcd_out = pcd_out.reshape(pcd.shape)
        
        pcds_list.append(pcd_out)
        transforms_list.append(T)
        transforms_inv_list.append(T_inv)

    color_augmentation = AugmentationSequential(
        # Color space augmentations
        K.ColorJitter(
            brightness=0.2,     # ±20% brightness change
            contrast=0.2,       # ±20% contrast change
            saturation=0.2,     # ±20% saturation change
            hue=0.1,           # ±10% hue change
            p=0.8              # 80% probability of applying
        ),
        
        # Gamma correction for exposure-like effects
        K.RandomGamma(
            gamma=(0.8, 1.2),   # Gamma range
            gain=(0.9, 1.1),    # Gain range
            p=0.5
        ),
        
        # HSV color space augmentations
        K.RandomHue(hue=(-0.1, 0.1), p=0.4),
        K.RandomSaturation(saturation=(0.8, 1.2), p=0.4),
        
        # Grayscale conversion (with probability)
        K.RandomGrayscale(p=0.1),
        
        # Posterization effect
        # K.RandomPosterize(bits=4, p=0.2),
        
        # Solarization effect
        # K.RandomSolarize(thresholds=0.1, additions=0.1, p=0.2),
        
        # Auto contrast
        K.RandomAutoContrast(p=0.3),
        
        # Random ordering and application
        random_apply=5,  # Randomly apply up to 10 transforms
        data_keys=["input"],  # Apply to input tensor
    )

    for i in range(batch_size):
        pcd_out_i = pcds_list[i].reshape(point_map.shape) # Reshape for masking
        # pcd_out_masked = pcd_out[mask]
        pcd_out_masked = pcd_out_i[mask]

        image_aug = color_augmentation(image)[0]
        colors_masked = image_aug[:, mask].T

        pcd_open3d = o3d.geometry.PointCloud()
        pcd_open3d.points = o3d.utility.Vector3dVector(pcd_out_masked.cpu().numpy())
        pcd_open3d.colors = o3d.utility.Vector3dVector(colors_masked.cpu().numpy())
        print(pcd_open3d)

        rendered_image, _, bg_mask = render_pcd_to_numpy_open3d(
            pcd_open3d,
            width=width, height=height,
            fx=fx, fy=fy, cx=cx, cy=cy,
            return_depth=True, return_mask=True,
            bg_color=(255, 255, 255)
        )

        images.append(torch.from_numpy(rendered_image).float().permute(2, 0, 1) / 255.0)
        masks.append(torch.tensor(~bg_mask))

        import cv2
        image_to_save = (image.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        cv2.imwrite(f"{i}.png", cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite("image.png", cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR))


    return {
        "images": torch.stack(images).to(pcd.device),
        "masks_gt": torch.stack(masks).to(pcd.device),
        "points_gt": torch.stack(pcds_list).to(pcd.device),
        "transforms": torch.stack(transforms_list).to(pcd.device),
        "transforms_inv": torch.stack(transforms_inv_list).to(pcd.device),
        "intrinsics": torch.stack(intrinsics).to(pcd.device)
    }
