
.. _tutorials_problem_modeling:


Modeling problems as DCOPs
==========================

What we have seen so far in previous tutorial may seems very theoretical
and it might not be obvious how DCOP could be used to solve
real-world problems.
pyDCOP is meant to be domain-independant, but we hope to convince you that
the DCOP algorithms implemented and studied with it can be applied to
concrete applications.

As a topic in the Multi-Agent System field,
DCOP are obviously best suited to problems that are distributed in nature.
They have been applied to a wide variety of applications, including
disaster evacuation :cite:`kopena_distributed_2008`,
radio frequency allocation :cite:`monteiro_channel_2012`,
recommendation systems :cite:`lorenzi_optimizing_2008`,
distributed scheduling :cite:`maheswaran_taking_2004`,
sensor networks :cite:`zhang_distributed_2005`,
intelligent environment :cite:`rust_using_2016`, :cite:`fioretto_multiagent_2017`,
smart grid :cite:`cerquides_designing_2015`,
etc.

When using a DCOP approach on a distributed problem, the first step is always
to cast your problem into an **optimization problem**
and to **identify your agents**.
In this tutorial, we will present one way of modelling a meeting scheduling
problem a a DCOP.



