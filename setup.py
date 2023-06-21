from setuptools import setup, find_packages

import os
import sys
import pathlib
# Get current directory
current_dir = os.getcwd()
# Get parent directory
parent_dir = os.path.join(current_dir, '..')
# Append parent directory to sys.path
sys.path.append(parent_dir)
cwd = pathlib.Path.cwd().parent

os.environ['RAW_TRAIN_DATA_PATH'] = cwd.joinpath('data/raw/train/EuansGuideData.xlsx')

os.environ['RAW_TEST_DATA_PATH'] = cwd.joinpath('data/raw/test/GoogleReviews')

os.environ['PROCESSED_DATA'] = cwd.joinpath('data/processed/aspect classification data')

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()


setup(
    name='venue-indoor-accessibility-pipeline',
    version='0.1.0',
    author='Mylène Brown-Coleman',
    author_email='m.j.c.browncoleman@student.vu.nl',
    description='This contains all of the packages used for the indoor venue accessibility pipeline.',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'my-pipeline=my_pipeline.pipeline:main',
            # define additional entry points here
        ]
    }
)