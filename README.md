# walking_human_trajectory_models
Two human-like locomotion models are impletemented in this repository : a model based on clothoid curves and a control optimal model.

To run the control optimal model, you need to install this version of the crocoddyl library : https://github.com/imaroger/crocoddyl/tree/mybranch.

To build the C++ code, this repository must be placed in a catkin workspace.

To cite this work in your academic research, please use the following bibtex lines:
```bibtex
@inproceedings{maroger20IROS,
  author={Maroger, Isabelle and Stasse, Olivier and Watier, Bruno},
  title={Walking Human Trajectory Models and Their Application to Humanoid Robot Locomotion},
  booktitle = {2020 IEEE/RSJ International Conference on Inteligent Robots and Systems (IROS)},
  year={2020}
}