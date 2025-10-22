import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fast-mikeio",
    version="0.0.1",
    author="Abbas Jazaeri",
    author_email="abja@dhigroup.com",
    description="A faster version of mikeio package for reading DHI MIKE output files.", 
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abja-dhi/fast-mikeio",
    packages=['fastmikeio'],
    install_requires=[
        'numpy',
        'mikecore',
        'matplotlib',
        'scipy',
        'cycler',
        'tqdm',
        ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

