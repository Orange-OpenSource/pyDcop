Distribution methods
====================

A distribution method is used to assign each computation to one agent, which will be responsible for hosting and running the computation.
In classical DCOP approaches, there is exactly one agent for each variable and most DCOP algorthims define one computation for each variable. 
In that case, the distribution of these computations is of course trivial. 
The `oneagent` distribution method replicate these traditional hypothesis in pydcop and might be enough 
if you do not care about distribution issues and simply want to develop or benchmark DCOP algorithms. 

However, these assumption do not hold on many real world problems ; 
agents typically maps to physical computers or devices and the number of these devices is not necessarily equals to the number of decision variables in the DCOP.
Moreover, some variables have a physical link to devices (in the sense, for example, that they model an action or decision of this particular device)
while some other variables might simply be used to model an abstract concept in the problem and have no real relationship with pysical devices.
For all these reason, the distribution of computations on agents is an interresting topic,
which is implemented in pydcop through distribution method.


A distribution method is used to assign each computation to one agent, which will be responsible for hosting and running the computation.
In classical DCOP approaches, there is exactly one agent for each variable and most DCOP algorthims computation define one computation for each variable. 
In that case, the distribution of these computations is of course trivial. 
The `oneagent` distribution method replicate these traditional hypothesis in pydcop and might be enough if you do not care about distribution issues and simply want to develop or benchmark DCOP algorithms. 

However, these assumption do not hold on many real world problems ; 
  agents typically maps to physical computers or devices and the number of these devices is not necessarily equals to the number of decision variables in the DCOP.
Moreover, some variables have a physical link to devices (in the sense, for example, that they model an action or decision of this particular device)
  while some other variables might simply be used to model an abstract concept in the problem and have no real relationship with pysical devices.
For all these reason, the distribution of computations on agents is an interresting topic,
  which is implemented in pydcop through distribution method.


Distribution methods are implemented in `pydcop.distribution`.
object.py defines objects that are used by all distribution methods (`Distribution` and `DistributionHints`).
A distribution methods is computes the allocation of a set computation to a set of agents. It should only take a computation graphs

To implement a new distribution method, one must: 

  * create a new module in `pydcop.distribution`, named after the distribution method
  * define the following methods in this file: 
    * `distribute`
    * `distribute_remove`
    * `distribute_add` 
