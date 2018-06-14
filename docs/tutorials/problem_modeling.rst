
.. _tutorials_problem_modeling:


Modeling problems as DCOPs
==========================

What we have seen so far in previous tutorials may seems very theoretical
and it might not be obvious how DCOP could be used to solve
real-world problems.
pyDCOP is meant to be domain-independent, but we hope to convince you that
the DCOP algorithms implemented and studied with it can be applied to
real applications.

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

When using a DCOP approach on a distributed problem, the first steps are always
to cast your problem into an **optimization problem**
and to **identify your agents**.
Then you can select, and probably benchmark, the best algorithm for the
settings of your problem.
In this tutorial, we will present one way of modelling a target tracking
problem as a DCOP.

Example: Target tracking problem
--------------------------------

Detecting and tracking mobile objects is a problem with many real applications
like surveillance and robot navigation, for example.
The goal of such a system is to detect foreign objects as quickly and as many
as possible.

The system is made of several sensor scattered in space.
Each sensor, for example a small Doppler effect sensor,
can only scan a fixed radius around itself at any given time;
it has to select with area it operates on.

In an open environment,
sensors used in tracking systems usually run on battery,
which means they must use as little energy as possible,
in order to increase the system operation's lifetime.
This includes switching them-self off and on whenever possible,
in a way that does not affect the system's detection performance.

These sensors are also lightweight devices
with limited memory and computation capability.
They communicate one with another through radio signals,
which may not be reliable.
Each sensor can only only communicates with neighboring sensors
and has no global information on the whole system.

The overall goal is thus to provide the **best detection** possible,
while **preserving energy** as much as possible.
To achieve this goal, sensors can act on several parameters:

* selecting which area to scan
* selecting when to switch on and off


Example: Target tracking DCOP model
-----------------------------------

Each sensor is controlled by one agent,
which decides the sector the sensor is scanning.
These agents coordinate in order to plan an efficient scanning strategy ;
this problem is actually a distributed planning system.

Let :math:`S = \{ S_1, ... S_n \}` be the set of **n sensors**.
Each agent :math:`S_i` can select the area
to scan among **k sectors** :math:`s_i = \{ s_i^1, ... s_i^k \}`.

The agents plan their action over a horizon :math:`T` made of
:math:`|T| = t` time slots.
For each time slot,
each agent :math:`S_i` has to select one
action : either scan one of it's :math:`s_i^j` sectors or sleep.

The :math:`s_i^k` are our **decision variables**,
whose value represents the sector scanned by a sensor at a given time.
These variables take their value from a domain
:math:`D = \{ 1, ... t\}` ;
when the variable :math:`s_i^k` takes the value :math:`t`,
it means that the sensor :math:`S_i` will scan the sector :math:`s_i^k`
during the time slot :math:`t`.

Of course, a sensor can only scan a single sector at any given time.
This can be modelled by defining a set of constraints :eq:`all_diff` ensuring
that two sectors from the same sensor cannot take the same value:

.. math::
  :label: all_diff

  \forall s_i^p, s_i^q \in s_i \times s_i, p \neq q
  \Rightarrow s_i^p  \neq s_i^q


For an efficient scanning process, we want to avoid several sensors scanning
simultaneously the same sector.
For this we define a function :math:`w` between a pair of sectors
:math:`s_i^p, s_j^q`
where :math:`w(s_i^p, s_j^q)` is the surface of the area common to these two
sectors.
Then we use this function to define constraints :eq:`conflict` between sectors,
where the cost of the constraints is this surface,
if the sensors of these two sector at scanning at the same time.


.. math::
  :label: conflict

  c(s_i^p, s_j^q) =
  \begin{cases}
    w(s_i^p, s_j^q) & \mathrm{if } s_i^p == s_j^q \\
    0 & \mathrm{otherwise}
  \end{cases}


With all these definitions, we can formulate the target tracking problem
as a DCOP
:math:`\langle \mathcal{A}, \mathcal{X}, \mathcal{D}, \mathcal{C}, \mu \rangle`
, where:

* :math:`\mathcal{A} =  \{ S_1, ... S_n \}` is the set of sensors;
* :math:`\mathcal{X} = \{ s_i^p\}, \quad S_i \in \mathcal{A}, \quad 0 \leq p \leq k`
  is the set of variables, for the k sectors of these n sensors;
* :math:`\mathcal{D} = \{0,...t\}` is the domain for these variable, made of the
  time slots in the forecasted horizon;
* :math:`\mathcal{C}` is the set of constraints over these variables, made of
  constraints :eq:`all_diff` and :eq:`conflict`;
* :math:`\mu` is a mapping function that assign each :math:`s_i^p` variable
  to the agent :math:`S_i`.

We can now use a DCOP algorithm to solve this problem in a distributed
manner.
Of course, the choice of the algorithm depends on the problem and the environment
characteristics; given that sensors have limited cpu and memory and that
the communication channel has a low bandwidth,
lightweight local search algorithm like DSA and MGM are good candidates.
The original article this model comes from, :cite:`zhang_analysis_2003`,
evaluates DSA and DBA and shows that,
if controlled properly, DSA is significantly superior to DBA,
finding better solutions with less computational cost
and communication overhead.


.. note:: In order to keep this tutorial short and relatively easy to read,
  the model presented here is a simplified version of the model exposed in
  :cite:`zhang_analysis_2003`.
  As you may have noticed, we do not take into account the possibility for an
  agent to *'sleep'* in order to save energy ; we only optimize the tracking
  to avoid inefficiencies.
  Moreover, the original model allows selecting several time slots for the same
  sector,
  which maps the target tracking problem to a multicoloring graph problem.






