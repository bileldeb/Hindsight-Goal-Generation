# Hindsight Goal Generation (HGG)

HGG fork to enable hgg on a pybullet mobile manipulator 


## Requirements
conda create -n hggenv python = 3.7
conda activate hggenv
pip install tensorflow==1.15
pip install protobuf==3.20.0
pip install BeautifulTable==0.7.0
pip install path/to/pandagym

## Running Commands

default env PandaMobilePickAndPlace-v3
episodes the number of hindsight goals

```bash
python train.py  
python train.py  --env=PandaReach-v3
python train.py  --env=PandaMobileReach-v3 --episodes=15 --timesteps=300

```