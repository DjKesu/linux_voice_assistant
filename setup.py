from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    name='WakeWordDetection',
    ext_modules=cythonize("wake_word_detection_lib.py"),
    zip_safe=False,
)