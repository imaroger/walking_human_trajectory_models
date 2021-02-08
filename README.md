# walking_human_trajectory_models


## Comparison of two human-like models 

Two human-like locomotion models are impletemented and compared in this repository : a model based on clothoid curves and an optimal control (OC) model.

To cite this work in your academic research, please use the following bibtex lines:
```bibtex
@inproceedings{maroger20IROS,
  author={Maroger, Isabelle and Stasse, Olivier and Watier, Bruno},
  title={Walking Human Trajectory Models and Their Application to Humanoid Robot Locomotion},
  booktitle = {2020 IEEE/RSJ International Conference on Inteligent Robots and Systems (IROS)},
  year={2020}
}
```

## Optimization of the OC model

An Inverse Optimal Control (IOC) scheme is implemented in this repository. It aims to optimize the weights of the OC model cost function in order to generate trajectories as close to human trajectories as possible.

To cite this work in your academic research, please use the following bibtex lines:
```bibtex
@inproceedings{maroger21,
  author={Maroger, Isabelle and Stasse, Olivier and Watier, Bruno},
  title={Inverse Optimal Control to Model Human Trajectories During Gait},
  booktitle = {submitted},
  year={2021}
}
```

## Installation

To run the OC model and the IOC scheme, you need to install this version of the crocoddyl library : https://github.com/imaroger/crocoddyl/tree/mybranch.

To build the C++ code, this repository must be placed in a catkin workspace.
