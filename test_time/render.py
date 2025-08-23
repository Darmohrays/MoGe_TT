import os
import logging
import warnings

import numpy as np
import open3d as o3d
from open3d.camera import PinholeCameraIntrinsic
from open3d.visualization.rendering import OffscreenRenderer, MaterialRecord

# Suppress verbose output
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

def render_pcd_to_numpy_open3d(
    pcd: o3d.geometry.PointCloud,
    width: int = 640,
    height: int = 480,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
    extrinsic: np.ndarray | None = None,
    point_size: float = 2.0,
    shader: str = "defaultUnlit",
    bg_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    use_gpu: bool = True,
):
    """
    Render a point cloud and return the color image, point cloud representation,
    and background mask.

    Args:
        pcd: The Open3D PointCloud object to render.
        width: The width of the rendered image.
        height: The height of the rendered image.
        fx: The focal length along the x-axis. Defaults to max(width, height).
        fy: The focal length along the y-axis. Defaults to max(width, height).
        cx: The principal point x-coordinate. Defaults to width / 2.0.
        cy: The principal point y-coordinate. Defaults to height / 2.0.
        extrinsic: The 4x4 extrinsic camera matrix (world to camera). Defaults to identity.
        point_size: The size of the points to be rendered.
        shader: The shader to use for rendering.
        bg_color: The RGB background color as a tuple of floats in [0, 1].
        use_gpu: Whether to attempt GPU acceleration.

    Returns:
        A tuple containing:
        - rendered_image (np.ndarray): (H, W, 3) uint8 RGB image.
        - rendered_pcd (np.ndarray): (H, W, 3) float32 array representing the point cloud.
                                     Background pixels are (0, 0, 0).
        - bg_mask (np.ndarray): (H, W) bool array, True for background pixels.
    """
    # Suppress verbose logging from Open3D
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    logging.getLogger('open3d').setLevel(logging.ERROR)
    warnings.filterwarnings('ignore', module='open3d')

    # ---- Set up camera intrinsics ----
    if fx is None or fy is None:
        fx = fy = max(width, height)
    if cx is None or cy is None:
        cx, cy = width / 2.0, height / 2.0
    intrinsic = PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    # ---- Set up camera extrinsics ----
    if extrinsic is None:
        extrinsic = np.eye(4, dtype=np.float32)

    # ---- Create and configure the renderer ----
    renderer = OffscreenRenderer(width, height)
    
    try:
        if use_gpu:
            # GPU-specific settings for performance
            renderer.scene.scene.enable_sun_light(False)
            renderer.scene.scene.enable_indirect_light(True)

        # Set background color
        renderer.scene.set_background(
            np.array([bg_color[0], bg_color[1], bg_color[2], 1.0], dtype=np.float32)
        )

        # Set up material properties
        mat = MaterialRecord()
        mat.shader = shader
        mat.point_size = float(point_size)

        # Add geometry to the scene
        renderer.scene.clear_geometry()
        renderer.scene.add_geometry("pcd", pcd, mat)

        # Set up the camera
        renderer.setup_camera(intrinsic, extrinsic)

        # ---- Render color image ----
        img_o3d = renderer.render_to_image()
        rendered_image = np.asarray(img_o3d)

        # ---- Render depth image to get Z coordinates ----
        depth_o3d = renderer.render_to_depth_image(z_in_view_space=True)
        depth_np = np.asarray(depth_o3d)

        # ---- Create background mask ----
        bg_mask = np.isinf(depth_np)

        # ---- Unproject depth image to 3D point cloud ----
        v, u = np.indices((height, width))
        Z = depth_np
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        points_camera_space = np.stack([X, Y, Z], axis=-1)
        points_camera_space[bg_mask] = 0.0

        # ---- Transform points from camera space to world space ----
        points_h = np.concatenate(
            [points_camera_space, np.ones((height, width, 1))], axis=-1
        )
        inv_extrinsic = np.linalg.inv(extrinsic)
        points_flat = points_h.reshape(-1, 4).T
        points_world_flat = inv_extrinsic @ points_flat
        rendered_pcd = points_world_flat.T.reshape(height, width, 4)[..., :3]

        return rendered_image, rendered_pcd.astype(np.float32), bg_mask

    finally:
        # Clean up renderer resources
        del renderer

def check_gpu_availability():
    """Check if GPU rendering is available and get detailed info."""
    print("=== GPU Rendering Diagnostics ===")
    
    # Check environment
    print(f"OPEN3D_USE_GPU: {os.environ.get('OPEN3D_USE_GPU', 'Not set')}")
    print(f"EGL_PLATFORM: {os.environ.get('EGL_PLATFORM', 'Not set')}")
    
    # Check system GPU info
    try:
        import subprocess
        # Check for NVIDIA
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("NVIDIA GPUs found:")
            print(result.stdout.strip())
        else:
            print("No NVIDIA GPUs detected")
    except:
        print("nvidia-smi not available")
    
    # Check OpenGL info
    try:
        result = subprocess.run(['glxinfo', '-B'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'OpenGL renderer' in line or 'OpenGL version' in line:
                    print(line.strip())
    except:
        print("glxinfo not available (install mesa-utils)")
    
    try:
        # Create a test renderer and capture any warnings
        print("\n=== Testing Open3D Rendering ===")
        test_renderer = OffscreenRenderer(64, 64)
        
        # Check if we can detect hardware acceleration
        print("- Renderer created successfully")
        
        # Try to render a simple scene
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector([[0, 0, 0]])
        pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
        
        mat = MaterialRecord()
        test_renderer.scene.add_geometry("test", pcd, mat)
        
        # Time the rendering to get a sense of performance
        import time
        start_time = time.time()
        test_img = test_renderer.render_to_image()
        render_time = time.time() - start_time
        
        print(f"- Render test: SUCCESS (took {render_time:.4f}s)")
        print(f"- Image shape: {np.asarray(test_img).shape}")
        
        del test_renderer
        return True
        
    except Exception as e:
        print(f"- Render test: FAILED ({e})")
        return False

# Example usage
if __name__ == "__main__":
    # Check GPU availability
    gpu_available = check_gpu_availability()
    
    # Create sample point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.randn(1000, 3))
    pcd.colors = o3d.utility.Vector3dVector(np.random.rand(1000, 3))
    
    # Render with GPU if available
    img = render_pcd_to_numpy_open3d(
        pcd, 
        width=1024, 
        height=1024,
        use_gpu=False
    )
    
    print(f"Rendered image shape: {img.shape}")
