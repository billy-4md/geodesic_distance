from setuptools import setup, Extension
import numpy as np

def get_extension():
    """
    Configure the C++ extension with appropriate flags and settings.
    
    Returns:
    --------
    Extension
        Configured extension for the geodesic_distance module
    """
    # Define source files
    sources = [
        './cpp/util.cpp',
        './cpp/geodesic_distance_2d.cpp',
        './cpp/geodesic_distance_3d.cpp',
        './cpp/geodesic_distance.cpp'
    ]
    
    # Configure the extension
    extension = Extension(
        'geodesic_distance',
        sources=sources,
        include_dirs=[
            np.get_include(),  # Include NumPy headers
            './cpp'           # Include local headers
        ],
        define_macros=[
            # Use new NumPy API
            ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
            # Define replacement for NPY_IN_ARRAY
            ('NPY_IN_ARRAY', 'NPY_ARRAY_IN_ARRAY')
        ],
        extra_compile_args=[
            '-std=c++11',     # Use C++11 standard
            '-O3',            # Enable high optimization
            '-fPIC'           # Position Independent Code
        ],
        language='c++'
    )
    
    return extension

setup(
    name='distance_transform',
    version='0.1.0',
    description='Geodesic distance transform implementation',
    ext_modules=[get_extension()],
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.7.0',
    ],
)