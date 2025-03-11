from setuptools import find_packages, setup

setup(
    name="tops",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "cloudpickle",
        "omegaconf",
        "easydict",
        "validators",
        "numpy",
        "torch",
        "tqdm",
        "Pillow",
        "tensorboard",
        "pyyaml",
    ],
    python_requires=">=3.10",
)
