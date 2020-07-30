from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='benchmarker',
    version='1.0.0',
    #url='https://github.com/mypackage.git',
    #author='Author Name',
    #author_email='author@gmail.com',
    #description='Description of my package',
    packages=find_packages(),
    install_requires=required
)