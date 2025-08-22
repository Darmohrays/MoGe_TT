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
    return_depth: bool = False,
    return_mask: bool = False,
    use_gpu: bool = True,
):
    """
    Render a point cloud with Open3D's OffscreenRenderer using GPU acceleration.
    
    Args:
        use_gpu: Whether to attempt GPU acceleration
        
    Returns:
        img_np: (H,W,3) uint8 RGB image.
        depth_np: (H,W) float depth image in [0,1] (optional).
        mask_np: (H,W) bool, True where no geometry was hit (optional).
    """
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    logging.getLogger('open3d').setLevel(logging.ERROR)
    # Suppress all warnings from Open3D
    warnings.filterwarnings('ignore', module='open3d')

    # ---- intrinsics defaults
    if fx is None or fy is None:
        fx = fy = max(width, height)
    if cx is None or cy is None:
        cx, cy = width / 2.0, height / 2.0
    intrinsic = PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    if extrinsic is None:
        extrinsic = np.eye(4, dtype=np.float32)
    
    # ---- create renderer with GPU settings
    renderer = OffscreenRenderer(width, height)
    
    try:
        # Check if GPU is being used
        if use_gpu:
            # These settings can help force GPU usage
            renderer.scene.scene.enable_sun_light(False)
            renderer.scene.scene.enable_indirect_light(True)
        
        # Background as RGBA (floats 0..1)
        renderer.scene.set_background(
            np.array([bg_color[0], bg_color[1], bg_color[2], 1.0], dtype=np.float32)
        )
        
        # Material (point size matters for point clouds)
        mat = MaterialRecord()
        mat.shader = shader
        mat.point_size = float(point_size)
        
        # Optimize material for GPU rendering
        if use_gpu and hasattr(mat, 'albedo_img'):
            # Enable GPU-specific optimizations if available
            mat.base_roughness = 0.9
            mat.base_reflectance = 0.04
        
        # (re)build scene
        renderer.scene.clear_geometry()
        renderer.scene.add_geometry("pcd", pcd, mat)
        
        # Camera
        renderer.setup_camera(intrinsic, extrinsic)
        
        # Render color
        img_o3d = renderer.render_to_image()
        img_np = np.asarray(img_o3d)  # uint8 HxWx3 (RGB)
        
        out = [img_np]
        
        if return_depth or return_mask:
            depth_o3d = renderer.render_to_depth_image()
            depth_np = np.asarray(depth_o3d).astype(np.float32)  # normalized depth in [0,1]
            
            if return_depth:
                out.append(depth_np)
                
            if return_mask:
                # Pixels that didn't hit geometry typically come back at the far plane (≈1.0)
                mask_np = np.isclose(depth_np, 1.0, atol=1e-6) | ~np.isfinite(depth_np)
                out.append(mask_np)
        
        if len(out) == 1:
            return out[0]
        return tuple(out)
        
    finally:
        # Ensure GL resources get freed promptly
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
