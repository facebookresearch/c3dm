# Copyright (c) Facebook, Inc. and its affiliates.

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="c3dm", # Replace with your own username
    version="1.0.0",
    author="Facebook AI Research",
    author_email="romansh@fb.com",
    description="""Code for the paper: Canonical 3D Deformer Maps: \
        Unifying parametric and non-parametric methods for \
        dense weakly-supervised category reconstruction\
    """,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/c3dm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA :: 10.1",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
    ],
    python_requires='>=3.6',
    install_requires=[
        "torch==1.5.1",
        "pytorch3d",
        "pyyaml>=5.3.1",
        "numpy>=1.17",
        "pillow>=1.7.2",
        "trimesh>=3.7.3",
        "matplotlib",
        "visdom>=0.1.8.9",
        "plotly>=4.8.1",
    ],
)