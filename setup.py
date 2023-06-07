import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="common-utils",
    version="0.0.1",
    author="Saeid Naderiparizi",
    description="A few useful functions for various ML projects",
    py_modules=["common_utils"],
    license='MIT License',
    python_requires=">=3.6",
    install_requires=[
            'wandb',
            'tqdm',
            'torch>=1.0.1',
        ],
)