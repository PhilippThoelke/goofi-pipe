from setuptools import setup, find_packages


with open("requirements.txt") as f:
    dependencies = f.read().splitlines()

setup(
    name="neurofeedback",
    version="0.0.1",
    packages=find_packages("neurofeedback"),
    install_requires=dependencies,
)
