from setuptools import setup, find_packages

setup(
    name="ml_toolkit",
    version="0.1",
    packages=find_packages(),
    install_requires = ["numpy"],
    author="Your Name",
    description="A simple ML toolkit for linear regression and data preprocessing"
)