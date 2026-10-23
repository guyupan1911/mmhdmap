"""Package metadata and runtime dependencies for mmhdmap."""

from pathlib import Path

from setuptools import find_packages, setup

PROJECT_ROOT = Path(__file__).resolve().parent

setup(
    name='mmhdmap',
    version='0.1.0',
    author='guyupan',
    author_email='guyupan1911@gmail.com',
    description='HD map perception components built on OpenMMLab 2.0',
    long_description=(PROJECT_ROOT / 'README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    packages=find_packages(include=['mmhdmap', 'mmhdmap.*']),
    package_data={'mmhdmap.visualization': ['assets/*.png']},
    python_requires='>=3.8',
    install_requires=[
        'mmengine>=0.10.7,<1.0.0',
        'numpy>=1.20.0',
        'pyquaternion>=0.9.9',
        'torch>=1.13.0',
    ],
    extras_require=dict(dev=['build>=1.0.0', 'pre-commit>=3.0.0']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
