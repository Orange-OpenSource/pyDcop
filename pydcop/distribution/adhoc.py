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


import logging
from random import choice, shuffle
from typing import Iterable

from collections import defaultdict

from pydcop.computations_graph.objects import ComputationGraph
from pydcop.dcop.objects import AgentDef
from pydcop.distribution.objects import Distribution, DistributionHints, \
    ImpossibleDistributionException

logger = logging.getLogger('distribution.adhoc')


"""
Ad-hoc distribution
This is an heuristic design for IJCAI 2016 paper.
The distribution respects the agents capacity, and thus requires an 
estimation function for the computation footprint, and honors so-called 
'hints'. 

"""


def distribute(computation_graph: ComputationGraph,
               agentsdef: Iterable[AgentDef],
               hints: DistributionHints=None,
               computation_memory=None,
               communication_load=None):
    """
    Generate a distribution for the dcop.
    This method uses a simple heuristic for distribution, with no guaranty of
    optimality.

    Even if a feasible distribution exists, this method is not warranted to
    find it.

    When using a dcop that represents an secp, given the correct 
    DistributionHint the same distribution should be generated that with the 
    adhoc secp distribution method.

    """
    if computation_memory is None:
        raise ImpossibleDistributionException('adhoc distribution requires '
                                              'computation_memory functions')

    agents = list(agentsdef)

    hints = DistributionHints() if hints is None else hints

    return _distribute_try(computation_graph, agents, hints,
                           computation_memory,
                           computation_graph)


def _distribute_try(computation_graph: ComputationGraph,
                    agents: Iterable[AgentDef],
                    hints: DistributionHints=None,
                    computation_memory=None,
                    communication_load=None,
                    attempt=0):

    agents_capa = {a.name: a.capacity for a in agents}
    # The distribution methods depends on the order used to process the node,
    # we shuffle them to test a new configuration when retry a distribution
    # after a failure
    nodes = list(computation_graph.nodes)
    shuffle(nodes)
    mapping = defaultdict(set)
    var_hosted = {}

    # Distribute owned computation variable on the corresponding agent.
    # For dcop build from an secp, this is the same thing as deploying the
    # light variable on the light devices, as we were doing before.
    for a in agents_capa:
        for c in hints.must_host(a):
            mapping[a].add(c)
            var_hosted.update({c: a})
            agents_capa[a] -= computation_memory(
                computation_graph.computation(c))

    # First mimic original secp adhoc behavior
    for n in nodes:
        if n.name in var_hosted:
            continue
        hostwith = hints.host_with(n.name)
        # secp models have a constraint that should be hosted on the same
        # agent than the variable of the model
        if len(hostwith) == 1 and n.type == 'FactorComputation' and \
            computation_graph.computation(hostwith[0]).type \
                == 'VariableComputation':

            dependent_var = [v.name for v in n.factor.dimensions]
            candidates = [a for a in agents_capa
                          if len(set(mapping[a]).intersection(
                                 dependent_var)) > 0]

            candidates.sort(key=lambda x: len(mapping[a]))
            if candidates:
                selected = candidates[0]
            else:
                selected = choice(list(agents_capa.keys()))

            mapping[selected].update({n.name, hostwith[0]})
            var_hosted[n.name] = selected
            var_hosted[hostwith[0]] = selected
            agents_capa[selected] -= computation_memory(n)

    for n in nodes:
        if n.name in var_hosted:
            continue
        footprint = computation_memory(n)
        # Candidates : hints only with enough capacity
        candidates = [(agents_capa[a], a) for a in hints.host_with(n.name)
                      if agents_capa[a] > footprint]
        # If no hinted agents has enough capacity, fall back to all agents
        if not candidates:
            candidates = [(c, a) for a, c in agents_capa.items()
                          if c > footprint]

        # Select the candidate that is already hosting the highest
        # number of computations sharing a link with this one.
        scores = []
        for capacity, a in candidates:
            count = 0
            for l in computation_graph.links_for_node(n.name):
                count += len([None for l_n in l.nodes
                              if l_n in mapping[a]])
            # The tuple is in this order so that we sort by score first,
            # and then by available capacity.
            scores.append((count, capacity, a))
        scores.sort(reverse=True)

        if scores:
            selected = scores[0][2]
            agents_capa[selected] -= footprint
        else:
            # Retry 3 times in case of failure, the nodes will be shuffled
            # every time, increasing the probability to find a feasible
            # distribution.
            if attempt > 2:
                raise ImpossibleDistributionException(
                    'Could not find feasible distribution after {} '
                    'attempts'.format(attempt))
            else:
                _distribute_try(computation_graph, agents, hints,
                                computation_memory, computation_graph,
                                attempt+1)

        mapping[selected].update({n.name})
        var_hosted[n.name] = selected

    return Distribution({a: list(mapping[a]) for a in mapping})


def distribute_remove(secp, current_distribution, removed_device):
    # TODO à implémenter !
    # FIXME : only take neighbors agents as variable ?
    raise NotImplementedError()


def distribute_add(secp, new_device, current_distribution,
                   connected_models=None):
    # TODO à implémenter !
    # FIXME : only take neighbors agents as variable ?
    raise NotImplementedError()
