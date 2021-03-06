# Adaptive and Recursive Period Control for communication efficient distributed SGD based optimization


## How to run this code

To run this project, it is necessary to install the OpenMPI packages and the packages in **requirement.txt** in a custom environment.
We recommend creating the environment inside the project folder. e.g., **AdaptiveRPC/env**

Setting up your local MPI cluster will also be necessary to run the code over multiple machines.
We followed the tutorial at the following link to install OpenMPI and configure our local cluster: https://feyziyev007.medium.com/how-to-install-openmpi-on-ubuntu-18-04-cluster-2fb3f03bdf61	

Once done, the strategy and other configurations can be adjusted in the **utils/options.py** file.								

## Simulate multiple workers on a single machine
The following command will simulate 5 machines(1 PS and 4 workers).

```
mpirun -np 5  PATH/AdaptiveRPC/env/bin/python PATH/AdaptiveRPC/main.py
```
PATH must be replaced with the absolute path to the project.

## Run with multiple machines

Let's say we have 5 machines respectively named: **host0,host1,host2,host3,host4**, sharing the folder **PATH/sharedfolder/** .
The project must be moved inside the shared folder, and the training can be executed as follows:

```
mpirun -np 5 --host host0,host1,host2,host3,host4 PATH/sharedfolder/AdaptiveRPC/env/bin/python PATH/sharedfolder/AdaptiveRPC/main.py
```

## Visualize logs with tensorboard

```
tensorboard --logdir PATH/sharedfolder/AdaptiveRPC/logs/
```