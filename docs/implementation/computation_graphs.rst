 
Computation Graphs
==================


Represent the graph of computations.
Implemented in pydcop.computation_graph
Always based on objects defined in pydcop.computation_graph.object

A module for a computation graph type typically contains

* class(es) representing the nodes of the graph (i.e. the computation),
  extending ComputationNode

* class representing the edges (extending Link)

* a class representing the graph

* a (mandatory) method  to build a computation graph from a Dcop object :

    def build_computation_graph(dcop: DCOP)-> ComputationPseudoTree:
