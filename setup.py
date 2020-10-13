import setuptools
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()
    
with open("requirements.txt", "r") as fh:
    requirements = fh.read()

setup(
    name='DRL',
    version = "0.0.1",
    author = "Anton Wiehe",
    author_email = "antonwiehe@gmail.com",
    description = ("Deep RL libary"),
    license = "MIT",
    keywords = "reinforcement learning, pytorch",
    url = "https://github.com/NotNANtoN/Deep-RL-Torch",
    packages = setuptools.find_packages(exclude=['tests', 'tests.*']),
    install_requires = requirements,
    long_description = long_description,
    long_description_content_type="text/markdown",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
