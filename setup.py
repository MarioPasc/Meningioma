from setuptools import setup, find_packages

setup(
    name='Meningioma',
    version='0.1',
    packages=find_packages(where='src'),  # Find packages inside 'src/Meningioma'
    package_dir={'': 'src'},  # 'src' is the base directory for the package
)
