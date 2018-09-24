Changelog
=========

Format based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)

pyDCOP v0.1.2 [Unreleased]
--------------------------

### Added
- New `--restart` flag on `agent` cli command.
- New `--version` global option on cli.
- `--graph` option may be omitted in `distribute` cli command, when `--algo`
 is given.
- `--output` is now supported by the `solve` cli command 
- Add a lot of documentation : usage, command line reference, etc. 
- `--delay` option for `solve` and `agent` cli command 
- termination detection in solve command: the command returns if all 
  computations have finished.
- Clean shutdown mechanism for orchestrator, agents, and messaging   
- support for periodic action at computation level.
- new @register decorator to register message handlers in computation classes.
- generator for graph coloring problem
- generator for meeting scheduling problems (PEAV model)
- generator for ising problems

### Fixed
- When stopping an agent, the ws-sever (for ui) was not closed properly.
- Issues causing delays when stopping the orchestrator.
- Invalid metrics containing management computations instead of agents.
- Avoid some crashes during metrics computations (when stopping the system 
  when metrics are not ready yet).
- Activity ratio computation was wrong.
- Bugs with end metric computations (cycle and time).
- Bug with solve and run command when collecting lots of metrics (would 
  not honor the timeout)   

## Modified
- domain type is now optional (in API and yaml DCOP format)
- agents can be given as a list or a dict in yaml
- the number and size of technical computations (mgt, discovery, etc.)  is not 
  output in metrics
- the port is now optional in the `agent`  and `orchestrator` cli commands 
- new mechanism for defining algorithms parameters   
- several periodic action can now be defined at agent'level 
  (only one action was possible before)
- agents, costs and routes are not serialized as empty map in yaml when they
  are not defined
- pydcop now requires python >= 3.6 

pyDCOP v0.1.0 - 2018-05-04
--------------------------

- First open source release.