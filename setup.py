import os
import glob
import shutil
from setuptools import find_packages, Command
import codecs

from numpy.distutils.core import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# set up automated versioning reading    
def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

liconvert_files = [
    'liconvert',
    'COPYING.txt',
]

liconvert_paths = []
for fname in liconvert_files:
    liconvert_paths.append(f"splendaq{os.sep}io{os.sep}_liconvert{os.sep}{fname}")


setup(
    name="splendaq",
    version=get_version(f'splendaq{os.sep}_version.py'),
    description="Data Acquisition for SPLENDOR",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="SPLENDOR Collaboration",
    url="https://github.com/splendor-collab/splendaq",
    license_files = ('LICENSE', ),
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'moku>=2.5',
        'h5py',
    ],
    data_files=[
        (f'splendaq{os.sep}io{os.sep}_liconvert{os.sep}', liconvert_paths),
    ],
)
