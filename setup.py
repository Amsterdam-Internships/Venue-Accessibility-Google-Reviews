from setuptools import setup, find_packages

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