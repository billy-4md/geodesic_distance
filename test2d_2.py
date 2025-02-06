import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
import geodesic_distance


def create_donut_image(size: tuple[int, int], outer_radius: int, inner_radius: int, 
                      center: tuple[int, int]) -> np.ndarray:
    """
    Create a binary image with a donut shape.
    
    Parameters:
    -----------
    size : tuple[int, int]
        Image size (height, width)
    outer_radius : int
        Outer radius of the donut
    inner_radius : int
        Inner radius of the donut
    center : tuple[int, int]
        Center coordinates (y, x)
        
    Returns:
    --------
    np.ndarray
        Binary image with donut (1 for donut, 0 for background)
    """
    y, x = np.ogrid[:size[0], :size[1]]
    dist_from_center = np.sqrt((y - center[0])**2 + (x - center[1])**2)
    
    # Créer les deux cercles
    outer_circle = (dist_from_center <= outer_radius)
    inner_circle = (dist_from_center <= inner_radius)
    
    # Créer le donut en soustrayant le petit cercle du grand
    donut = (outer_circle & ~inner_circle).astype(np.uint8)
    return donut


def find_leftmost_point(binary_image: np.ndarray) -> tuple[int, int]:
    """
    Find the leftmost point of a binary shape.
    
    Parameters:
    -----------
    binary_image : np.ndarray
        Binary image where shape is marked with 1s
        
    Returns:
    --------
    tuple[int, int]
        Coordinates of leftmost point (y, x)
    """
    y_coords, x_coords = np.where(binary_image == 1)
    leftmost_idx = np.argmin(x_coords)
    return y_coords[leftmost_idx], x_coords[leftmost_idx]


def compute_seed_distance(binary_image: np.ndarray, seed_point: tuple[int, int]) -> np.ndarray:
    """
    Compute distance from seed point for all points inside the shape.
    
    Parameters:
    -----------
    binary_image : np.ndarray
        Binary image with shape
    seed_point : tuple[int, int]
        Starting point coordinates (y, x)
        
    Returns:
    --------
    np.ndarray
        Distance map from seed point (float32)
    """
    y, x = np.ogrid[:binary_image.shape[0], :binary_image.shape[1]]
    dist_from_seed = np.sqrt((y - seed_point[0])**2 + (x - seed_point[1])**2)
    # Normalize distances to [0, 1] range inside the shape
    dist_inside = dist_from_seed * binary_image
    max_dist = np.max(dist_inside[binary_image > 0])
    dist_inside = (dist_inside / max_dist).astype(np.float32)
    return dist_inside


def test_geodesic_distance2d():
    """
    Test geodesic distance computation inside a donut shape.
    Compares fast marching and raster scan methods.
    """
    # Create binary image with donut
    image_size = (256, 256)
    center = (128, 128)
    outer_radius = 50
    inner_radius = 25  # La moitié du rayon externe
    
    # Generate donut mask
    donut_mask = create_donut_image(image_size, outer_radius, inner_radius, center)
    
    # Create seed point at leftmost point of donut
    seed_point = find_leftmost_point(donut_mask)
    S = np.zeros_like(donut_mask, np.uint8)
    S[seed_point[0], seed_point[1]] = 1
    
    # Create distance-based gradient from seed point
    I = compute_seed_distance(donut_mask, seed_point)
    
    # Set high values outside the donut (make sure it's float32)
    I[donut_mask == 0] = 1e10
    I = I.astype(np.float32)  # Ensure float32 type
    
    # Compute distances using both methods
    t0 = time.time()
    D1 = geodesic_distance.geodesic2d_fast_marching(I, S)
    t1 = time.time()
    D2 = geodesic_distance.geodesic2d_raster_scan(I, S)
    
    dt1 = t1 - t0
    dt2 = time.time() - t1
    print(f"Runtime (s) fast marching: {dt1:.3f}")
    print(f"Runtime (s) raster scan:   {dt2:.3f}")
    
    # Mask the distances to show only inside the donut
    D1[donut_mask == 0] = np.nan
    D2[donut_mask == 0] = np.nan
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(donut_mask, cmap='gray')
    plt.plot(seed_point[1], seed_point[0], 'ro')
    plt.axis('off')
    plt.title('Input Image with Seed Point')
    
    plt.subplot(1, 3, 2)
    plt.imshow(D1, cmap='hot')
    plt.colorbar(label='Distance')
    plt.axis('off')
    plt.title('Fast Marching')
    
    plt.subplot(1, 3, 3)
    plt.imshow(D2, cmap='hot')
    plt.colorbar(label='Distance')
    plt.axis('off')
    plt.title('Raster Scan')
    
    plt.tight_layout()
    plt.show()
    
    # Debug information
    print(f"D1 range: {np.nanmin(D1):.3f} to {np.nanmax(D1):.3f}")
    print(f"D2 range: {np.nanmin(D2):.3f} to {np.nanmax(D2):.3f}")


if __name__ == '__main__':
    test_geodesic_distance2d()