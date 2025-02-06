import geodesic_distance
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nib


def load_nifty_volume_as_array(filename: str) -> np.ndarray:
    """
    Load a NIfTI volume and convert it to a numpy array.
    
    Parameters:
    -----------
    filename : str
        Path to the NIfTI file (.nii or .nii.gz)
    
    Returns:
    --------
    np.ndarray
        3D numpy array with shape [D, H, W]
        The data is transposed from NIfTI format [W, H, D] to [D, H, W]
    """
    img = nib.load(filename)
    # Use get_fdata() instead of deprecated get_data()
    data = img.get_fdata()
    # Transpose from [W, H, D] to [D, H, W]
    data = np.transpose(data, [2, 1, 0])
    return data

def save_array_as_nifty_volume(data: np.ndarray, filename: str) -> None:
    """
    Save a numpy array as a NIfTI volume.
    
    Parameters:
    -----------
    data : np.ndarray
        3D numpy array with shape [D, H, W]
    filename : str
        Output filename for the NIfTI file
    """
    # Transpose from [D, H, W] to [W, H, D] for NIfTI format
    data = np.transpose(data, [2, 1, 0])
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, filename)

def geodesic_distance_3d(I: np.ndarray, S: np.ndarray, lamb: float, iter: int) -> np.ndarray:
    """
    Get 3D geodesic distance by raster scanning.
    
    Parameters:
    -----------
    I : np.ndarray
        Input image volume
    S : np.ndarray
        Binary image where non-zero pixels are used as seeds
    lamb : float
        Weighting between 0.0 and 1.0
        - if lamb=0.0, returns spatial euclidean distance without considering gradient
        - if lamb=1.0, distance is based on gradient only without using spatial distance
    iter : int
        Number of iterations for raster scanning
    
    Returns:
    --------
    np.ndarray
        3D array containing geodesic distances
    """
    return geodesic_distance.geodesic3d_raster_scan(I, S, lamb, iter)

def test_geodesic_distance3d():
    """
    Test function to compare fast marching and raster scan methods on 3D data.
    Loads a 3D volume, computes geodesic distances using both methods,
    and visualizes results on a 2D slice.
    """
    # Load and preprocess data
    I = load_nifty_volume_as_array("data/img3d.nii")
    I = np.asarray(I, np.float32)
    I = I[18:38, 63:183, 93:233]
    
    # Create seed point
    S = np.zeros_like(I, np.uint8)
    S[10, 60, 70] = 1
    
    # Compute distances using both methods
    t0 = time.time()
    D1 = geodesic_distance.geodesic3d_fast_marching(I, S)
    t1 = time.time()
    D2 = geodesic_distance_3d(I, S, 1.0, 4)
    dt1 = t1 - t0
    dt2 = time.time() - t1
    
    print(f"runtime(s) fast marching {dt1:.3f}")
    print(f"runtime(s) raster scan   {dt2:.3f}")

    # Normalize and save results
    D1 = (D1 * 255 / D1.max()).astype(np.uint8)
    save_array_as_nifty_volume(D1, "data/image3d_dis1.nii")

    D2 = (D2 * 255 / D2.max()).astype(np.uint8)
    save_array_as_nifty_volume(D2, "data/image3d_dis2.nii")
    
    I = (I * 255 / I.max()).astype(np.uint8)
    save_array_as_nifty_volume(I, "data/image3d_sub.nii")

    # Visualize results
    I_slice = I[10]
    D1_slice = D1[10]
    D2_slice = D2[10]
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(I_slice, cmap='gray')
    plt.plot([70], [60], 'ro')
    plt.axis('off')
    plt.title('Input Image')
    
    plt.subplot(1, 3, 2)
    plt.imshow(D1_slice)
    plt.axis('off')
    plt.title('Fast Marching')
    
    plt.subplot(1, 3, 3)
    plt.imshow(D2_slice)
    plt.axis('off')
    plt.title('Raster Scan')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    test_geodesic_distance3d()