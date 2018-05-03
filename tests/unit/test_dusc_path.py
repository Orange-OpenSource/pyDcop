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


from queue import Queue, Empty
from typing import List, Dict, Tuple

import networkx as nx


# nx.draw_spectral(G)
#
# plt.show()

"""
The algorithm work in round
at each round the budget is increased 

Messages :

replicate
  * budget 
  * spent 
  * closed
  * queue
  * visited 

replicated
  * closed
  * visited 
  * hosts (merge with visited / closed?) 
  * queue  


Ideas for a proof:

We want to prove that 
 * the algorithm always terminates in finite graph
 * with no limit on the budget, we will eventually visit all nodes
 * agents are always visited on increasing cost order 

Invariants

 * the cost of an agent in the queue is the minimum budget needed to reach 
 new agents. we might not reach new agent with that budget but with less we 
 are sure we will NOT discover any new agents
 
 * at each round the budget is stricly increasing
 
 * during a round, we never visit the same agent twice.  

"""


def cheapest_path_to(target: str, paths: Dict[Tuple, float])\
        -> Tuple[float, Tuple]:
    c = float('inf')
    path = ()
    for p in paths:
        if p[-1] == target:
            return paths[p], p
    return c, path


def path_starting_with(start_path: Tuple, paths: Dict[Tuple, float]) \
        -> List[Tuple[float, Tuple]]:
    l = len(start_path)
    found = [(c, p[l:]) for p, c in paths.items()
             if p[:l] == start_path]
    return sorted(found)


def route(a: str, b: str) -> float:
    return G[a][b]['cost']


def replicate(rq_path: Tuple, budget: float, spent: float,
              paths: Dict[Tuple, float],
              visited):
    print('replicate', rq_path, budget, spent, paths, visited)

    current = rq_path[-1]
    if rq_path in paths:
        paths.pop(rq_path)
    if current not in visited:
        visited.append(current)

    n_paths = path_starting_with(rq_path, paths)
    if n_paths:
        for cost, p in n_paths:
            if cost - spent <= budget:
                msg_queue.put(('replicate', rq_path + (p[0],),
                               budget - route(current, p[0]),
                               spent + route(current, p[0]),
                               paths, visited))
                return

    neighbors = ((n, route(current, n), rq_path + (n,))
                 for n in G.neighbors(current)
                 if n not in visited)
    for n, r, p in neighbors:
        cheapest, cheapest_path = cheapest_path_to(n, paths)
        if cheapest > spent + r:
            if cheapest_path in paths:
                paths.pop(cheapest_path)
            paths[p] = spent + r
        else:
            print('CHEAPER PATH KNOWN TO ', p)

    previous = rq_path[-2]
    msg_queue.put(('replicated', rq_path,
                   budget + route(previous, current),
                   spent - route(previous, current),
                   paths, visited))


def replicated(rq_path: Tuple, budget: float, spent: float,
               paths: Dict[Tuple, float], visited: List):
    print('replicated ', budget, spent, rq_path, paths, visited)

    *_, current, sender = rq_path
    initial_path = rq_path[:-1]

    n_paths = path_starting_with(initial_path, paths)
    for cost, p in n_paths:

        if p[0] != sender and cost-spent <= budget:
            msg_queue.put(('replicate', initial_path + (p[0],),
                           budget - route(current, p[0]),
                           spent + route(current, p[0]),
                           paths, visited))
            return

    # Could not send to any neighbor
    if len(rq_path) >= 3:
        previous = rq_path[-3]
        msg_queue.put(('replicated', initial_path,
                       budget + route(current, previous),
                       spent - route(current, previous),
                       paths, visited))
        return

    # no reachable candidate path and no ancestor to go back,
    # we are back at the start node: increase the budget
    try:
        budget = min(c for c in paths.values())
        print('  ## visited : ', visited, ' with budget', budget)
        msg_queue.put(('replicate', (current,), budget, 0, paths, visited))
        print('  ## BUDGET INCREASE ', budget)
    except ValueError:
        print('No costs left : finished', visited)
        msg_queue.put(('result', visited))
        return


msg_queue = Queue()


def init(start: str):

    global msg_queue
    msg_queue = Queue()

    # initialize paths with start node neighbors
    paths = {(start, n): G[start][n]['cost']
             for n in G.neighbors(start)}
    budget = min(c for c in paths.values())
    visited = [start]

    msg_queue.put(('replicate', (start,), budget, 0, paths, visited))

    while True:
        try:
            msg = msg_queue.get(block=False)
            if msg[0] == 'replicate':
                replicate(*msg[1:])
            elif msg[0] == 'replicated':
                replicated(*msg[1:])
            if msg[0] == 'result':
                visited = msg[1]
                break
        except Empty:
            break

    return visited

G = None


def test_linear():
    # Simple linear uniform cost graph
    global G
    G = nx.Graph()
    G.add_edge('A', 'B', cost=1)  # default edge data=1
    G.add_edge('B', 'C', cost=1)  # specify edge data
    G.add_edge('C', 'D', cost=1)
    visited = init('A')
    print('Visited : ',  visited)

    assert visited == ['A', 'B', 'C', 'D']
    print('\n\n')


def test_one_branch():
    global G
    G = nx.Graph()
    G.add_edge('A', 'B', cost=1)
    G.add_edge('B', 'C', cost=1)
    G.add_edge('B', 'D', cost=1)
    G.add_edge('C', 'E', cost=1)
    visited = init('A')

    print('Visited : ',  visited)

    # both order are valid
    assert visited == ['A', 'B', 'D', 'C', 'E'] or \
        visited == ['A', 'B', 'C', 'D', 'E']
    print('\n\n')


def test_no_cheapest_neighbor_first():
    # Example demonstrating that we should not always visit the cheapest
    # neighbor first
    # Visit order should be
    # C B F D E
    global G
    G = nx.Graph()
    G.add_edge('A', 'B', cost=3)
    G.add_edge('A', 'C', cost=1)
    G.add_edge('B', 'F', cost=1)
    G.add_edge('C', 'E', cost=5)
    G.add_edge('C', 'D', cost=4)
    visited = init('A')

    print('Visited : ',  visited)

    assert visited == ['A', 'C', 'B', 'F', 'D', 'E']
    print('\n\n')


def test_one_loop():
    global G
    G = nx.Graph()
    G.add_edge('A', 'B', cost=1)
    G.add_edge('A', 'C', cost=2)
    G.add_edge('B', 'C', cost=1)
    visited = init('A')
    print('Visited : ',  visited)

    assert visited == ['A', 'B', 'C'] or visited == visited == ['A', 'C', 'B']
    print('\n\n')


def test_one_loop_2():
    global G
    G = nx.Graph()
    G.add_edge('A', 'B', cost=2)
    G.add_edge('B', 'C', cost=1)
    G.add_edge('B', 'D', cost=2)
    G.add_edge('C', 'E', cost=2)
    G.add_edge('D', 'E', cost=2)
    G.add_edge('E', 'F', cost=5)
    G.add_edge('D', 'G', cost=1)
    visited = init('A')
    print('Visited : ',  visited)

    # order between E and G does not matter, the cost of the path from A to
    # them is the same (5)
    assert visited == ['A', 'B', 'C', 'D', 'E', 'G', 'F'] or \
        visited == ['A', 'B', 'C', 'D', 'G', 'E', 'F']
    print('\n\n')


def test_complex():
    global G
    G = nx.Graph()
    G.add_edge('A', 'B', cost=1)
    G.add_edge('A', 'C', cost=2)

    G.add_edge('B', 'D', cost=3)
    G.add_edge('B', 'E', cost=1)

    G.add_edge('C', 'E', cost=1)
    G.add_edge('C', 'F', cost=4)

    G.add_edge('D', 'H', cost=1)
    G.add_edge('D', 'I', cost=3)

    G.add_edge('E', 'I', cost=1)
    G.add_edge('E', 'G', cost=2)

    visited = init('A')
    print('Visited : ',  visited)

    # order between E and C does not matter, same thing for D and G
    assert visited[0] == 'A'
    assert visited[1] == 'B'
    assert (visited[2] == 'C' and visited[3] == 'E') or \
           (visited[3] == 'C' and visited[2] == 'E')
    assert visited[4] == 'I'
    assert (visited[5] == 'D' and visited[6] == 'G') or \
           (visited[6] == 'D' and visited[5] == 'G')
    assert visited[7] == 'H'
    assert visited[8] == 'F'

    # To be exact, we should not need a budget higher than 6
    # assert visited == ['A', 'B', 'C', 'E', 'I', 'D', 'G', 'H', 'F'] or \
    #        ['A', 'B', 'E', 'C', 'I', 'D', 'G', 'H', 'F']

    # test_linear()
# test_one_branch()
# test_no_cheapest_neighbor_first()
# test_one_loop()
