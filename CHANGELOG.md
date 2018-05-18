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
- Add a lot of documentation : usage, command line reference, etc. 


### Fixed
- When stopping an agent, the ws-sever (for ui) was not closed properly.
- Issues causing delays when stopping the orchestrator.


pyDCOP v0.1.0 - 2018-05-04
--------------------------

- First open source release.