# deformable_ros

RoboCraft: Learning to See, Simulate, and Shape Elasto-Plastic Object with Graph Networks

The controller is based on ROS Noetic and [Polymetis](https://facebookresearch.github.io/fairo/polymetis/). It's especially useful if you're working with a Franka Panda robot arm and a Franka hand gripper. If your robot setup is different, this codebase won't work out of the box, but there might be some useful scripts. Feel free to contact me at hshi74@stanford.edu if you have any questions.

## Overview

**[Project Page](http://hxu.rocks/robocraft/) |  [Paper](https://arxiv.org/pdf/2205.02909.pdf)**

## Prerequisites
- Linux (Tested on Ubuntu 20.04)
- ROS Noetic
- [Polymetis](https://facebookresearch.github.io/fairo/polymetis/).
- Conda

## Getting Started

### Setup
- Install the prerequisites.
```bash
# clone the repo into the src folder of your catkin workspace
git clone https://github.com/hshi74/deformable_ros.git

# cd to the catkin workspace and build it
catkin_make

# create the conda environment
conda env create -f polymetis.yml
conda activate polymetis
```

### For all the following bash or python scripts, you will need to modify certain hyperparameters (like directories) before you run them.

### Generate Data
Run `python src/random_explore.py` in one terminal and `python src/collect_dy_data.py` in another terminal.
