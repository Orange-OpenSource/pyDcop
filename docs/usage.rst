Usage
=====

pyDCOP can be used as a command-line application or as an library, using its
API. 

.. toctree::
   :maxdepth: 1

   usage/cli_ref
   usage/fileformat_ref
   usage/api_ref


Most common use-cases
---------------------

Running a DCOP locally
^^^^^^^^^^^^^^^^^^^^^^

Solving a static DCOP::

  pydcop -t 4  -v 3 solve --algo maxsum  tests/instances/graph_coloring1.yaml


Running a DCOP (continuously)::

  pydcop -t 360 --log log.conf run -a maxsum
         --scenario scenario.yaml
         -d dist_dcop_100.yaml
         -m thread
         dcop_100.yaml


Running a DCOP on several computers 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* orchestator

::

  pydcop -v 3  orchestrator -a maxsum -d adhoc  tests/instances/graph_coloring1.yaml

* agents, on each of the machines, run

::

  pydcop -v 3 agent -n a<num> -p 900<num>
                    --orchestrator <orchestrator_ip>
                    --uiport 10001


Other commands
^^^^^^^^^^^^^^

Generating DCOP instances

Distributing a DCOP

Distributing replicas of computations 

