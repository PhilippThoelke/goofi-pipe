from setuptools import find_packages, setup

with open("requirements.txt") as f:
    dependencies = f.read().splitlines()

setup(
    name="goofi-pipe",
    version="1.0.0",
    packages=find_packages("goofi"),
    install_requires=dependencies,
)
