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


from collections import defaultdict
from typing import Dict, List, Iterable, Union


class Distribution(object):
    """
    This object is a convenient representation of a distribution with
    methods for querying it (has_computation, agent_for, etc.)

    Parameters
    ----------
    mapping: mapping: Dict[str, List[str]]
        A dict  agent name: [computation names]. Basic validity checks are
        performed to ensure that the same computation is not hosted on
        several agents.

    """
    def __init__(self, mapping: Dict[str, List[str]]):
        # { agent_name : {list of comp_name]}
        self._mapping = mapping  # type: Dict[str, List[str]]
        # {comp_name : agent_name }
        self._computation_agent = {}  # type: Dict[str, str]
        for a in self._mapping:
            for v in self._mapping[a]:
                if v in self._computation_agent:
                    raise ValueError(
                        'Inconsistent distribution : several agents hosting '
                        'computation {} : {} and {}'
                        .format(v, a, self._computation_agent[v]))
                self._computation_agent[v] = a

    @property
    def agents(self) -> List[str]:
        """
        Agents used in this distribution

        Returns
        -------
        agents list:
            The list of the names of agents used in this distribution.

        """
        return list(self._mapping)

    @property
    def computations(self) -> List[str]:
        """
        Distributed computations

        Returns
        -------
        computations list:
            A list containing the names of the computations distributed in this
            distribution.
        """
        return [c for l in self._mapping.values() for c in l]

    def mapping(self) -> Dict[str, List[str]]:
        """
        The distribution represented as a dict.

        Returns
        -------
        Dict[str, List[str]]:
            A dict associating a list of computation names to each agent name.
        """
        return dict(self._mapping)

    def agent_for(self, computation: str)-> str:
        """
        Agent hosting one given computation
        Parameters
        ----------
        computation: str
            a computation's name

        Returns
        -------
        str:
            the name of the agent hosting this computation.
        """
        if computation not in self._computation_agent:
            raise KeyError('No computation {} in this distribution'.format(
                computation))
        else:
            return self._computation_agent[computation]

    def computations_hosted(self, agent: str)-> List[str]:
        """
        Computations hosted on an agent.

        If the agent is not used in the distribution (its name is not known),
        returns an empty list.

        Parameters
        ==========
        agent: str
            the name of the agent

        Returns
        =======
        List[str]:
            The list of computations hosted by this agent.
        """
        try:
            return list(self._mapping[agent])
        except KeyError:
            return []

    def has_computation(self, computation: str):
        """

        Parameters
        ----------
        computation: str
            computation name

        Returns
        -------
        Boolean:
            True if this computation is part of the distribution
        """
        return computation in self._computation_agent

    def host_on_agent(self, agent: str, computations: List[str]):
        """
        Host several computations on an agent.

        Modify the distribution by adding computations to be hosted on agent.
        If this agent name is unknown, it is added, otherwise the list of
        computations is added to the computations already hosted by this agent.

        Parameters
        ----------
        agent: str
            an agent name
        computations: List[str]
            A list of computation names

        """
        for v in computations:
            if v in self._computation_agent:
                raise ValueError('Computation {} is already hosted on agent '
                                 '{}'. format(v, self._computation_agent[v]))
        if agent not in self._mapping:
            self._mapping[agent] = computations
        else:
            self._mapping[agent] += computations
        for v in computations:
            self._computation_agent[v] = agent

    def is_hosted(dist, computations: Union[str, Iterable[str]]):
        """
        Indicates if some computations are hosted.

        This methods does not care on which agent the computations are hosted.

        Parameters
        ----------
        computations: List[str]
            A list of computation names

        Returns
        -------
        Boolean:
            True if all computations are hosted.

        """
        if isinstance(computations, str):
            computations = [computations]
        for computation in computations:
            try:
                dist.agent_for(computation)
            except KeyError:
                return False
        return True

    def __str__(self):
        return 'Distribution({})'.format(self.mapping())

    def __repr__(self):
        return 'Distribution({})'.format(self.mapping())

    def __eq__(self, other):
        if type(other) != Distribution:
            return False
        if self.mapping() == other.mapping():
            return True
        return False


class DistributionHints(object):

    def __init__(self, must_host=None, host_with=None):
        """
        
        :param must_host: {map {agent_name : list of computation names} 
        :param host_with: map {name: [list of names]} where names are all 
        computation names
        """
        self._must_host = must_host

        if host_with is None:
            self._host_with = {}
        else:
            host_with_tmp = defaultdict(lambda: set())
            for i in host_with:
                host_with_tmp[i].update(host_with[i])
                for j in host_with[i]:
                    s = {i}.union(host_with[i])
                    s.remove(j)
                    host_with_tmp[j].update(s)

            self._host_with = {n: list(host_with_tmp[n])
                               for n in host_with_tmp}

    def must_host(self, agt_name: str)-> List[str]:
        """
        :param agt_name: The name of the agent.
         
        :return: A (possibly empty) list of constraints and variables names 
        whose computation must be 
        hosted on the agent named `agt_name`. 
        """
        if self._must_host is not None:
            return self._must_host[agt_name][:] if agt_name in \
                                                   self._must_host else []
        else:
            return []

    def host_with(self, name: str)-> List[str]:
        if name in self._host_with:
            return self._host_with[name][:]
        else:
            return list()


class ImpossibleDistributionException(Exception):

    pass