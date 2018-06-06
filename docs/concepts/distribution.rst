
.. _concepts_distribution:

DCOP Computations distribution
==============================


Before running the DCOP, the computation must be deployed on agents.
We name *distribution* the task of assigning each computation to one agent,
which will be responsible for hosting and running the computation.
In classical DCOP approaches, there is exactly one agent for each variable
and most DCOP algorithms define one computation for each variable.
In that case, the distribution of these computations is of course trivial.
The **oneagent** distribution replicates these traditional hypothesis
in pyDCOP and might be enough if you do not care about distribution issues and
simply want to develop or benchmark DCOP algorithms.

However, these assumptions do not hold on many real world problems.
Agents typically maps to physical computers or devices
and the number of these devices is not necessarily equal
to the number of decision variables in the DCOP.
Moreover, some variables have a physical link to devices
(in the sense, for example, that they model an action or decision
of this particular device)
while some other variables might simply be used to model
an abstract concept in the problem and have no real relationship
with physical devices.
For all these reasons, the distribution of computations on agents is
an interesting topic, which is implemented in pyDCOP through distribution
methods.
See :cite:`Rust2017` for a more detailled explanation of the deployment
and distribution of computations in a DCOP.
