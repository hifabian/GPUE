import setuptools

setuptools.setup(
    name="GPUE",
    version="alpha",
    author="Lee J. O\'Riordan, James R. Schloss",
    author_email="loriordan@gmail.com",
    description="GPUE Python bindings",
    long_description="GPUE Python bindings",
    url="https://github.com/gpue-group/gpue",
    packages=setuptools.find_packages(),
    package_data={'': ['_PyGPUE.*.so'],},
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
