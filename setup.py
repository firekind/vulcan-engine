from setuptools import setup, find_packages

setup(
    name="vulcan-engine",
    version="0.0.1",
    packages=find_packages(exclude=['blank']),
    install_requires=[
        'numpy==1.16.4',
        'PyOpenGL==3.1.3b1',
        'PyOpenGL-accelerate==3.1.3b1',
        'glfw==1.8.1',
        'Pillow==8.2.0',
    ],
    author='Shyamant'
)
