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


from setuptools import setup, find_packages


with open('README.md', 'r') as f:
    README = f.read()

# Basic dependencies, required to run pyDCOP:
deps = [
        'pulp',
        'numpy',
        'networkx',
        'pyyaml',
        'requests',
        'websocket-server'
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
    'sphinx_rtd_theme'
]

# Required to install dev dependencies with pip
#    pip install -e .[test]
extras = {
    'test': test_deps,
    'doc': doc_deps
}


setup(
    name='pydcop',
    version='0.1.0',
    description='Several dcop algo implementation',
    long_description=README,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",

        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.5",

        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    author='Pierre Rust',
    author_email='pierre.rust@orange.com',

    keywords=['dcop', 'MAS'],

    install_requires=deps,
    tests_require=test_deps,
    extras_require=extras,

    scripts=[
        'pydcop/pydcop',
        'pydcop/dcop.py'
    ],

    packages =find_packages()
)
