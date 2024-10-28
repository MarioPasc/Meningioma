# setup.py
from setuptools import setup, find_packages

setup(
    name="Meningioma",  
    version="0.1.0",  
    packages=find_packages(where="src"),  
    package_dir={"": "src"},  
    install_requires=[
        "opencv-python-headless",  
        "matplotlib",
        "scikit-image",
        "scipy",
        "pynrrd",
        "SciencePlots",
        "pandas",
        "openpyxl",
        "ipykernel"
    ],

    author="Pascual Gonzalez, Mario",
    author_email="mario.pg02@gmail.com",
    description="A Python library that contains all the funtions used to perform experiments on neuroimaging data and generate a Meningioma dataset.",
    url="https://github.com/MarioPasc/Meningioma",  
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
