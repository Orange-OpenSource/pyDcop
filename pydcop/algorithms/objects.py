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


from typing import Iterable, Dict, Any

from pydcop.utils.simple_repr import SimpleRepr, simple_repr, from_repr


class AlgoDef(SimpleRepr):
    """
    An AlgoDef represent a given dpop algorithm with all parameter needed to
    run it. These parameter generally depend on a specific algorithm (e.g.
    variant A, B or C for DSA and damping factor for maxsum).

    """

    def __init__(self, algo: str, mode: str='min', **params) -> None:
        """

        :param algo: name of the algorithm. It must be the name of a module
        in the `pydcop.algorithms` package.

        :param mode: min of max
        :param params: keywords argument for algo-specific configuration and
        parameters
        """
        self._algo = algo
        self._mode = mode
        self._params = params  # type: Dict[str, Any]

    @property
    def algo(self) -> str:
        return self._algo

    @property
    def mode(self) -> str:
        return self._mode

    def param_names(self) -> Iterable[str]:
        return self._params.keys()

    def param_value(self, param: str) -> Any:
        return self._params[param]

    @property
    def params(self)-> Dict:
        return dict(self._params)

    def _simple_repr(self):
        r = super()._simple_repr()
        r['params'] = simple_repr(self._params)
        return r

    @classmethod
    def _from_repr(cls, r):
        params =  r['params']
        del r['params']
        args = {k: from_repr(v) for k, v in r.items()
                if k not in ['__qualname__', '__module__']}
        algo =  cls(**args, **params)
        return algo

    def __str__(self):
        return 'AlgoDef({})'.format(self.algo)

    def __repr__(self):
        return 'AlgoDef({}, {}, {})'.format(self.algo, self.mode, self._params)

    def __eq__(self, other):
        if type(other) != AlgoDef:
            return False
        if self.algo != other.algo or self.mode != other.mode:
            return False
        if self._params != other._params:
            return False
        return True