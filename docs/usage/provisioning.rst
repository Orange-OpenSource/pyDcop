
.. _usage_provisioning:

Provisioning pyDCOP on many machines
====================================


When developing on pyDCOP, if you want to host your agent on several different
machines, you generally need to deploy your changes often in order to test them.
This can be a pain if you do it manually (copying pyDCOP's new development
version on all machines, installing it, etc. ) and is error prone.
For this reasons it is advised to automate this deployment. We provide an
ansible playbook for this, along with some advices.

General principle
-----------------

In the following sections, we use the term **controller** to name the
computer that is used to initiate and drive the installation of pyDCOP on
other machines, which are called **agent-machine** (as they will be used to run
pyDCOP agents).

When provisioning a set of machines, the **controller** will
contacts each **agent-machines** (using ssh) and make sure pyDCOP and all its
dependencies are installed on it.

This guide uses `ansible <https://www.ansible.com/>`_
(tested with version 2.5.3) , thus you must install
ansible on the controller, which requires python. If python is already
installed on your controller machine you can simply use::

  sudo pip install ansible

See `The official installation guide for more details <https://docs.ansible
.com/ansible/latest/installation_guide/intro_installation.html>`_

.. warning:: Using windows for the Ansible controller is not officially
  supported. However you can `run it under the Windows Subsystem for Linux (WSL)
  <http://docs.ansible.com/ansible/latest/user_guide/windows_faq
  .html#can-ansible-run-on-windows>`_
  or `with Cygwin <http://www.oznetnerd.com/installing-ansible-windows/>`_.
  Another option is to use a linux virtual machine as a controller.

Installing without internet connection
--------------------------------------

In many cases (tutorials, lectures, demos) it is necessary to install pyDCOP
(including its dependencies ) on machines that may not have internet access.
For this purpose, we recommend using *cache services* on the controller, that
will provide the necessary packages to agent-machines even though the internet
repositories cannot be reached.

To be sure to have all required packages cached on the controller machine,
you must install at least one agent machine while the controller has internet
access.
The controller will then cache locally all packages used during the
installation and
you will later be able to deploy new agent-machines with no internet access,
even on the controller.

Local cache for apt
^^^^^^^^^^^^^^^^^^^

Some system packages (e.g. ``glpk`` ) need to be installed with apt-get,
we use the local cache apt repository
`ACNG <https://www.unix-ag.uni-kl.de/~bloch/acng/>`_,
hosted on the controller computer.

Install on controller::

    sudo apt-get install apt-cacher-ng

Check that ACNG is online:  http://localhost:3142/acng-report.html

See `the official documentation <http://xmodulo.
com/apt-caching-server-ubuntu-debian.html>`_ for more details.

Local cache for pip
^^^^^^^^^^^^^^^^^^^

We run `devpi <https://www.devpi.net/>`_, which acts as a proxy-cache, on the
controller::

  pip install -q -U devpi-server
  pip install -q -U devpi-web
  devpi-server --start --init
  devpi-server --stop

Note: the `devpi` server must be started every time we need to need to perform
a deployment. Once finished you may stop it::

  devpi-server --start --host=0.0.0.0
  ...
  devpi-server --stop

You can check that devpi is online : http://127.0.0.1:3141/


Agent-machines
--------------

You can use any computer as an agent-machine, however we recommend using
linux-based machines (Macs should work and windows can work too, but this
guide might require some tweaking).
The main requirement for ansible is that your agent-machine must have an ssh
server. 
The command ``sudo`` must be available and your user must be able to use 
(i.e. be in the sudoer list, 
simply adding it to the sudo group usually does the trick
``adduser dcop sudo``)

As we generally do not have lots of machines available, agents-machines
are generally implemented using Virtual Machines (VM) or cheap single-board
computers like the Raspberry Pi.

Virtual Machines
^^^^^^^^^^^^^^^^

Using Virtual Machines (VM) allows you to run several agent-machines
on a single server.
We use VirtualBox to run our VM, but you can use any hypervisor.

Create a base linux (debian or ubuntu is recommended, as the playbook assumes
packages can be installed with ``apt``) and duplicate it to create as many VM
as required. Do not forget to enable the ssh server on your VM !

Make sure you configure the network mode on the hypervisor in a way that
allows connecting to servers running on the VMs.
The details depends on your hypervisor, with VirtualBox you generally want
to use the "bridged adaptater" mode (and not NAT, which the default).


Raspberry Pis
^^^^^^^^^^^^^

Single board computers like the Raspberry Pi are very good candidates for
agent-machines : they are cheap, run linux and are powerfull enough for
pyDCOP.

You should use the `standard raspbian distribution <https://www.raspberrypi
.org/downloads/>`_.
The only modification is to make sure that the ssh
server is enabled your Raspberry Pi. This can be done using the ``
raspi-config`` utility or by creating a file named ``ssh`` in the boot
partition of the SD card, which can be done easily after preparing the SD
card on another computer and is the best option if you run your Pis without
keyboards.
See the official `documentation <https://www.raspberrypi
.org/documentation/remote-access/ssh/>`_
for more details


Configuration
-------------

Once you have your agent-machine (and you have noted their IP address), you
must edit the ansible host configuration file. An example is provided in
pyDCOP's git repository (in ``pyDcop/provisioning/ansible/hosts-conf.yaml``)


.. warning:: Be careful, agent names given in the host configuration file
  **must match** the names given in the dcop yaml definition, and their IP
  address must be set to the IP address assigned to the corresponding VMs or
  physical machine.

TODO: explain how to use avahi to make your agent-machine automatically
discoverable.


Deploying with ansible
----------------------

Once you have properly configured your host file, you can simply run
ansible-playbook to apply the operations on all your agent-machines.
The playbook is in ``pyDcop/provisioning/ansible/``::

    ansible-playbook --inventory hosts-conf.yaml pydcop-playbook.yml

If the process fails on some machines, you can safely restart it as ansible
keeps track of the progress.

You can also run the playbook on a subset of the hosts defined in your 
configuration file, by using the ``--limit`` option:

    ansible-playbook --inventory hosts-conf.yaml --limit a2 pydcop-playbook.yml
