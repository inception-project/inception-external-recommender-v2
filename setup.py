# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from setuptools import find_packages, setup

NAME = "galahad"
DESCRIPTION = "Machine learning model server that can predict AND train"
HOMEPAGE = "https://github.com/inception-project/inception-external-recommender-v2"
EMAIL = "git@mrklie.com"
AUTHOR = "Jan-Christoph Klie"
REQUIRES_PYTHON = ">=3.7.0"

install_requires = [
    "fastapi==0.65.*",
    "uvicorn[standard]==0.13.*",
    "sortedcontainers>==2.4.*",
    "joblib==1.0.*",
    "datasets==1.6.*",
    "filelock==3.0.*",
    "requests~=2.26.0",
    "gradio==2.7.0",
    "nltk==3.6.7"
]

test_dependencies = ["codecov", "pytest", "pytest-cov", "datasets"]

dev_dependencies = ["black", "isort"]

doc_dependencies = ["sphinx", "sphinx-autodoc-typehints", "sphinx-rtd-theme"]

spacy_dependencies = [
    "spacy>=3.0.*",
]

sklearn_dependencies = ["scikit-learn>=0.24.*"]

contrib_dependencies = []
contrib_dependencies.extend(spacy_dependencies)
contrib_dependencies.extend(sklearn_dependencies)

extras = {
    "test": test_dependencies,
    "dev": dev_dependencies,
    "doc": doc_dependencies,
    "contrib": contrib_dependencies,
    "spacy": spacy_dependencies,
    "sklearn": sklearn_dependencies,
}

here = os.path.abspath(os.path.dirname(__file__))

with open("README.md", "r") as fh:
    long_description = fh.read()

# Load the package"s __version__.py module as a dictionary.
about = {}
with open(os.path.join(here, "galahad", "__version__.py")) as f:
    exec(f.read(), about)


# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown; charset=UTF-8",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=HOMEPAGE,
    packages=find_packages(exclude="tests"),
    keywords="slurm",
    project_urls={
        "Bug Tracker": "https://github.com/inception-project/inception-external-recommender-v2/issues",
        "Documentation": "https://github.com/inception-project/inception-external-recommender-v2",
        "Source Code": "https://github.com/inception-project/inception-external-recommender-v2",
    },
    install_requires=install_requires,
    test_suite="tests",
    tests_require=test_dependencies,
    extras_require=extras,
    include_package_data=True,
    license="Apache License 2.0",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Human Machine Interfaces",
        "Topic :: Text Processing :: Linguistic",
    ],
)
