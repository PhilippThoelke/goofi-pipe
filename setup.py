from setuptools import find_packages, setup

with open("requirements.txt") as f:
    dependencies = f.read().splitlines()

setup(
    name="neurofeedback",
    version="0.0.1",
    packages=find_packages(),
    install_requires=dependencies,
)
