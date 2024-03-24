from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define the extension module with necessary include directories for NumPy
extensions = [
    Extension("kmeans_cy", ["kmeans_cy.pyx"], include_dirs=[np.get_include()])
]

# Use cythonize on the extension object
setup(
    ext_modules=cythonize(extensions, annotate=True),  # Pass 'annotate=True' here
)
