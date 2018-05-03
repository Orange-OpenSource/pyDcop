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


from typing import Dict, List

from collections import defaultdict

from pydcop.algorithms import ComputationDef
from pydcop.infrastructure.computations import Message


class ReplicaDistribution(object):
    """
    Simply a convenient representation of the distribution of replica on agents

    """

    def __init__(self, mapping: Dict[str, List[str]]):
        """
        Basic
        :param mapping: map computation -> list of agents hosting a replica
        for this computation.
        """
        self._mapping = mapping  # type: Dict[str, List[str]]
        self._agent_replicas = \
            defaultdict(lambda: [])  # type: Dict[str, List[str]]

        for c in self._mapping:
            for a in self._mapping[a]:

                if c in self._agent_replicas[a]:
                    raise ValueError('Agent {} is hosting several replica '
                                     'for {}'.format(a, c))
                self._agent_replicas[a].append(c)

    def replicas_on(self, agt: str, raise_on_unknown=False):
        try:
            return list(self._agent_replicas[agt])
        except KeyError as ke:
            if raise_on_unknown:
                raise ke
            return []

    def agents_for_computation(self, computation: str):
        return list(self._mapping[computation])
