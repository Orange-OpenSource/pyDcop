# BSD-3-Clause License
#
# Copyright 2017 Orange
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.



# Basic dependencies, required to run pyDCOP:
deps = [
        'pulp',
        'numpy',
        'networkx',
        'pyyaml',
        'requests',
        'websocket-server',
        'tqdm',
    ]

# Extra dependencies, used to run tests
test_deps = [
    'coverage',
    'pytest',
    'mypy'
]

# Extra dependencies, used to generate docs
doc_deps = [
    'sphinx',
    'sphinx_rtd_theme',
    ' sphinxcontrib-bibtex'
]

# Required to install dev dependencies with pip
#    pip install -e .[test]
extras = {
    'test': test_deps,
    'doc': doc_deps
}


from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(path.join(here, 'pydcop', 'version.py'), encoding='utf-8') as f:
    exec(f.read())

setup(
    name='pydcop',
    version=__version__,
    description='Several dcop algo implementation',

    long_description=long_description,
    long_description_content_type='text/markdown', 

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",

        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.5",    

        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    author='Pierre Rust (Orange)',
    author_email='pierre.rust@orange.com',

    keywords=['dcop', 'MAS'],

    install_requires=deps,
    tests_require=test_deps,
    extras_require=extras,

    scripts=[
        'pydcop/pydcop',
        'pydcop/dcop_cli.py'
    ],

    packages =find_packages(),

    project_urls={
        'Documentation':  'http://pydcop.readthedocs.io',
        'Source': 'https://github.com/Orange-OpenSource/pyDcop',
        'Bug Reports': 'https://github.com/Orange-OpenSource/pyDcop/issues'
    }    
)
