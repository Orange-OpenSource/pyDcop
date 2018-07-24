
.. _tutorials_agent_gui:


Agent's GUI
===========


Start the Web UI for agent. 

  cd ~/pyDcop-ui/dist
  python3 -m http.server 4001

Run a dynamic DCOP, using the extra using the following command::

  pydcop --log log.conf orchestrator \
         --algo maxsum --algo_params damping:0.9 \
         --distribution heur_comhost \
         --scenario scenario_2.yaml \
         graph_coloring_20.yaml

  pydcop agent --names a000 a001 a002 a003 a004 a005 a006 a007 a008 a009 a010 \
    a011 a012 a013 a014 a015 a016 a017 a018 a019 a020 \
     -p 9001 --orchestrator 127.0.0.1:9000 --uiport 10001