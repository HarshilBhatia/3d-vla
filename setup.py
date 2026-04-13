from setuptools import setup, find_packages


setup(
    name="3d_flowmatch_actor",
    version="0.1.0",
    packages=find_packages(where="."),   # or specify subfolders
    package_dir={"": "."},               # interpret current dir as top-level
    install_requires=[
        "hydra-core>=1.3.0",
    ],
)
