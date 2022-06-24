
from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import setup
from setuptools import find_packages


def _requires_from_file(filename):
    return open(filename).read().splitlines()


with open("README.md", "r") as f:
    long_description = f.read()


DISTNAME = 'functionalanova'
AUTHOR = 'Yu Kiyokawa'
AUTHOR_EMAIL = 'y-kiyokawa@bird-initiative.com'
LICENSE = 'MIT License'
VERSION = '0.1.0'
KEYWORDS = 'anova'
PYTHON_REQUIRES = ">=3.7"
DISTDIR = 'functionalANOVA'
# DATA = 'configs'

setup(
    name=DISTNAME,
    version=VERSION,
    license=LICENSE,
    description="A Python Package for ANOVA.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url="",
    keywords=KEYWORDS,
    python_requires=PYTHON_REQUIRES,
    packages=[s.replace(DISTDIR, DISTNAME) for s in find_packages('.')],
    package_dir={DISTNAME: DISTDIR},
    # package_data={DISTDIR: [f'{DATA}/*.dat']},
    py_modules=[splitext(basename(path))[0]
                for path in glob(f'{DISTDIR}/*.py')],
    install_requires=_requires_from_file('requirements.txt'),
)
