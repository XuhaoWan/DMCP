import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    dependencies = fh.readlines()

setuptools.setup(
    name="DMCP",
    packages=setuptools.find_packages(exclude=["tests"]),
    version="0.1.2",
    author="Xuhao Wan",
    author_email="xhwanrm@whu.edu.cn",
    description="DMCP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/XuhaoWan/DMCP",
    python_requires=">=3.7",
    install_requires=dependencies,
    license="GNU",
    classifiers=[
        "License :: OSI Approved :: GNU  License",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Development Status :: 4 - Beta",
    ],
)
