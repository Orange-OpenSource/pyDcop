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

from pydcop.computations_graph.factor_graph import build_computation_graph
from pydcop.dcop.yamldcop import load_dcop
from pydcop.distribution.oneagent import distribute
from pydcop.infrastructure import synchronous_single_run
from pydcop.infrastructure.agents import deploy_on_local_agents

logging.basicConfig(level=logging.DEBUG)
logging.info('MaxSum Smart Lighting using MultipleComputationAgent')


dcop_yaml = """

name: graph coloring
description: "This example comes from the 10_3_2_0.4_r0 from the PDCSdata
             data set used in the article :
             R.T. Maheswaran, J.P. Pearce, M. Tambe. “Distributed algorithms
             for DCOP: a graphical-game-based approach.” In PDCS, 2004.
             Downloaded from http://teamcore.usc.edu/dcop/"
objective: min

domains:
  colors:
    values: [R, G]
    type: 'color'

variables:
  v0:
    domain: colors
    cost_function: 0 if v0 else 0
    noise_level: 0.01    
  v1:
    domain: colors
    cost_function: 0 if v1 else 0
    noise_level: 0.01    
  v2:
    domain: colors
    cost_function: 0 if v2 else 0
    noise_level: 0.01    
  v3:
    domain: colors
    cost_function: 0 if v3 else 0
    noise_level: 0.01    
  v4:
    domain: colors
    cost_function: 0 if v4 else 0
    noise_level: 0.01    
  v5:
    domain: colors
    cost_function: 0 if v5 else 0
    noise_level: 0.01    
  v6:
    domain: colors
    cost_function: 0 if v6 else 0
    noise_level: 0.01    
  v7:
    domain: colors
    cost_function: 0 if v7 else 0
    noise_level: 0.01    
  v8:
    domain: colors
    cost_function: 0 if v8 else 0
    noise_level: 0.01    
  v9:
    domain: colors
    cost_function: 0 if v9 else 0
    noise_level: 0.01    

constraints:
  diff_5_7: 
    type: intention
    function: 1 if v5 == v7 else 0
  diff_1_7: 
    type: intention
    function: 1 if v1 == v7 else 0
  diff_3_5: 
    type: intention
    function: 1 if v3 == v5 else 0
  diff_1_5: 
    type: intention
    function: 1 if v1 == v5 else 0
  diff_4_7: 
    type: intention
    function: 1 if v4 == v7 else 0
  diff_2_5: 
    type: intention
    function: 1 if v2 == v5 else 0
  diff_0_4: 
    type: intention
    function: 1 if v0 == v4 else 0
  diff_1_9: 
    type: intention
    function: 1 if v1 == v9 else 0
  diff_5_8: 
    type: intention
    function: 1 if v5 == v8 else 0
  diff_6_9: 
    type: intention
    function: 1 if v6 == v9 else 0
  diff_0_6: 
    type: intention
    function: 1 if v0 == v6 else 0
  diff_1_2: 
    type: intention
    function: 1 if v1 == v2 else 0
  diff_6_8: 
    type: intention
    function: 1 if v6 == v8 else 0
  diff_0_8: 
    type: intention
    function: 1 if v0 == v8 else 0
  diff_6_7: 
    type: intention
    function: 1 if v6 == v7 else 0
  diff_0_9: 
    type: intention
    function: 1 if v0 == v9 else 0
  diff_4_8: 
    type: intention
    function: 1 if v4 == v8 else 0
  diff_3_8: 
    type: intention
    function: 1 if v3 == v8 else 0
  diff_2_3: 
    type: intention
    function: 1 if v2 == v3 else 0
  diff_7_9: 
    type: intention
    function: 1 if v7 == v9 else 0

agents:
  a1:
    capacity: 100
  a2:
    capacity: 100
  a3:
    capacity: 100
  a4:
    capacity: 100
  a5:
    capacity: 100
  a6:
    capacity: 100
  a7:
    capacity: 100
  a8:
    capacity: 100
  a9:
    capacity: 100
  a10:
    capacity: 100
  a11:
    capacity: 100
  a12:
    capacity: 100
  a13:
    capacity: 100
  a14:
    capacity: 100
  a15:
    capacity: 100
  a16:
    capacity: 100
  a17:
    capacity: 100
  a18:
    capacity: 100
  a19:
    capacity: 100
  a20:
    capacity: 100
  a21:
    capacity: 100
  a22:
    capacity: 100
  a23:
    capacity: 100
  a24:
    capacity: 100
  a25:
    capacity: 100
  a26:
    capacity: 100
  a27:
    capacity: 100
  a28:
    capacity: 100
  a29:
    capacity: 100
  a30:
    capacity: 100
    
"""

dcop = load_dcop(dcop_yaml)

cg = build_computation_graph(dcop)

mapping = distribute(cg, dcop.agents)

agents = deploy_on_local_agents(cg, mapping)

results, _, _ = synchronous_single_run(agents, 10)

print("FINISHED ! " + str(results))