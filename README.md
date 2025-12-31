# **Self-contained implementation for paper "Variational OOD State Correction for Offline Reinforcement Learning".**

DASP realize OOD state correction via the specifically designed variational model.

The implementation is based on [RORL](https://github.com/YangRui2015/RORL). Thanks a lot for their contribution.

## 1. Requirements

To install the required dependencies:

1. Install the MuJoCo 2.1 engine, which can be downloaded from [here](https://mujoco.org/download).

2. Install Python packages in the requirement file, [d4rl](https://github.com/rail-berkeley/d4rl) and `dm_control`. The commands are as follows:
  
```bash
conda create -n dasp python=3.8.5
conda activate dasp
pip install --no-cache-dir -r requirements.txt

git clone https://github.com/rail-berkeley/d4rl.git
cd d4rl
# Note: remove lines including 'dm_control' in setup.py of d4rl
pip install -e .
```

## 2. Usage DASP Experiments for MuJoCo Gym with pretrained VAE model

### 2.1 Training
```bash
python -m scripts.sac --env_name [ENVIRONMENT] --num_qs [NUMBEROFQ] --load_config_type 'benchmark' --model_path [model path] --exp_prefix DASP
```
'ENVIRONMENT': e.g., walker2d-medium-replay-v2

'NUMBEROFQ': e.g., 7 or 10.

### 2.2 Evaluation
#### To evaluate trained agents in standard environments, run
```bash
python -m scripts.sac --env_name [ENVIRONMENT] --num_qs [NUMBEROFQ] --eval_no_training --model_path [model path] \
--load_path [load path] --exp_prefix eval_DASP
```
'load path':  e.g., ~/offline_best.pt.

'NUMBEROFQ': e.g., 7 or 10.

#### To evaluate trained agents in adversarial environments, run
```bash
python -m scripts.sac --env_name [ENVIRONMENT] --eval_no_training --load_path [model path] \
--load_path [load path] --eval_attack  --eval_attack_mode [mode] --eval_attack_eps [epsilon]  --exp_prefix eval_DASP
```
'mode': 'random, action_diff, min_Q'.

'epsilon': [0.0, 0.3].


[//]: # (## Citation)

