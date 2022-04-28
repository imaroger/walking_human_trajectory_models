# walking_human_trajectory_models


## Comparison of two human-like models 

Two human-like locomotion models are impletemented and compared in this repository : a model based on clothoid curves and an optimal control (OC) model.

A video of this work is available on : <https://www.youtube.com/watch?v=ZmAJzs6VDlw>

To cite this work in your academic research, please use the following bibtex lines:
```bibtex
@INPROCEEDINGS{maroger2020IROS,
  author={Maroger, I. and Stasse, O. and Watier, B.},
  booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Walking Human Trajectory Models and Their Application to Humanoid Robot Locomotion}, 
  year={2020},
  volume={},
  number={},
  pages={3465-3472},
  doi={10.1109/IROS45743.2020.9341118}}

```

## Optimization of the OC model

An Inverse Optimal Control (IOC) scheme is implemented in this repository. It aims to optimize the weights of the OC model cost function in order to generate trajectories as close to human trajectories as possible.

To cite this work in your academic research, please use the following bibtex lines:
```bibtex
@article{maroger2021CMBBE,
author = {Isabelle Maroger and Olivier Stasse and Bruno Watier},
title = {Inverse optimal control to model human trajectories during locomotion},
journal = {Computer Methods in Biomechanics and Biomedical Engineering},
volume = {25},
number = {5},
pages = {499-511},
year  = {2022},
publisher = {Taylor & Francis},
doi = {10.1080/10255842.2021.1962311}}
```

## Installation

To run the OC model and the IOC scheme, you need to install this version of the crocoddyl library : https://github.com/imaroger/crocoddyl/tree/mybranch.

To build the C++ code, this repository must be placed in a catkin workspace.
