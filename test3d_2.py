import numpy as np
import time
import napari
import geodesic_distance


def create_hollow_sphere(size: tuple[int, int, int], outer_radius: int, inner_radius: int,
                        center: tuple[int, int, int]) -> np.ndarray:
    """
    Create a binary volume with a hollow sphere shape.
    
    Parameters:
    -----------
    size : tuple[int, int, int]
        Volume size (depth, height, width)
    outer_radius : int
        Outer radius of the sphere
    inner_radius : int
        Inner radius of the sphere
    center : tuple[int, int, int]
        Center coordinates (z, y, x)
        
    Returns:
    --------
    np.ndarray
        Binary volume with hollow sphere (1 for sphere, 0 for background)
    """
    z, y, x = np.ogrid[:size[0], :size[1], :size[2]]
    dist_from_center = np.sqrt((z - center[0])**2 + 
                              (y - center[1])**2 + 
                              (x - center[2])**2)
    
    # Create the two spheres
    outer_sphere = (dist_from_center <= outer_radius)
    inner_sphere = (dist_from_center <= inner_radius)
    
    # Create the hollow sphere by subtracting the inner sphere from the outer one
    hollow_sphere = (outer_sphere & ~inner_sphere).astype(np.uint8)
    return hollow_sphere


def find_leftmost_point_3d(binary_volume: np.ndarray) -> tuple[int, int, int]:
    """
    Find the leftmost point of a binary shape in 3D.
    
    Parameters:
    -----------
    binary_volume : np.ndarray
        Binary volume where shape is marked with 1s
        
    Returns:
    --------
    tuple[int, int, int]
        Coordinates of leftmost point (z, y, x)
    """
    z_coords, y_coords, x_coords = np.where(binary_volume == 1)
    leftmost_idx = np.argmin(x_coords)
    return z_coords[leftmost_idx], y_coords[leftmost_idx], x_coords[leftmost_idx]


def compute_seed_distance_3d(binary_volume: np.ndarray, 
                           seed_point: tuple[int, int, int]) -> np.ndarray:
    """
    Compute distance from seed point for all points inside the shape.
    
    Parameters:
    -----------
    binary_volume : np.ndarray
        Binary volume with shape
    seed_point : tuple[int, int, int]
        Starting point coordinates (z, y, x)
        
    Returns:
    --------
    np.ndarray
        Distance map from seed point (float32)
    """
    z, y, x = np.ogrid[:binary_volume.shape[0], 
                       :binary_volume.shape[1], 
                       :binary_volume.shape[2]]
    dist_from_seed = np.sqrt((z - seed_point[0])**2 + 
                            (y - seed_point[1])**2 + 
                            (x - seed_point[2])**2)
    
    # Normalize distances to [0, 1] range inside the shape
    dist_inside = dist_from_seed * binary_volume
    max_dist = np.max(dist_inside[binary_volume > 0])
    dist_inside = (dist_inside / max_dist).astype(np.float32)
    return dist_inside


def test_geodesic_distance3d():
    """
    Test geodesic distance computation inside a hollow sphere.
    Compares fast marching and raster scan methods.
    Visualizes results using Napari.
    """
    # Create binary volume with hollow sphere
    volume_size = (128, 128, 128)
    center = (64, 64, 64)
    outer_radius = 40
    inner_radius = 20
    
    # Generate hollow sphere mask
    sphere_mask = create_hollow_sphere(volume_size, outer_radius, inner_radius, center)
    
    # Create seed point at leftmost point of sphere
    seed_point = find_leftmost_point_3d(sphere_mask)
    S = np.zeros_like(sphere_mask, np.uint8)
    S[seed_point[0], seed_point[1], seed_point[2]] = 1
    
    # Create distance-based gradient from seed point
    I = compute_seed_distance_3d(sphere_mask, seed_point)
    
    # Set high values outside the sphere
    I[sphere_mask == 0] = 1e10
    I = I.astype(np.float32)
    
    # Compute distances using both methods
    t0 = time.time()
    D1 = geodesic_distance.geodesic3d_fast_marching(I, S)
    t1 = time.time()
    D2 = geodesic_distance.geodesic3d_raster_scan(I, S, 1.0, 4)
    
    dt1 = t1 - t0
    dt2 = time.time() - t1
    print(f"Runtime (s) fast marching: {dt1:.3f}")
    print(f"Runtime (s) raster scan:   {dt2:.3f}")
    
    # Mask the distances to show only inside the sphere
    D1[sphere_mask == 0] = np.nan
    D2[sphere_mask == 0] = np.nan
    
    # Create viewer and add layers
    viewer = napari.Viewer()
    
    # Add the original mask
    viewer.add_image(
        sphere_mask,
        name='Hollow Sphere',
        colormap='gray',
        opacity=0.5
    )
    
    # Add the seed point
    viewer.add_points(
        [seed_point],
        name='Seed Point',
        size=5,
        face_color='red'
    )
    
    # Add the distance maps
    viewer.add_image(
        D1,
        name='Fast Marching',
        colormap='hot',
        visible=False
    )
    
    viewer.add_image(
        D2,
        name='Raster Scan',
        colormap='hot'
    )
    
    # Debug information
    print(f"D1 range: {np.nanmin(D1):.3f} to {np.nanmax(D1):.3f}")
    print(f"D2 range: {np.nanmin(D2):.3f} to {np.nanmax(D2):.3f}")
    
    napari.run()


if __name__ == '__main__':
    test_geodesic_distance3d()