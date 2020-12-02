# CLR research framework

Two components: 
- [`RLLib 1.0.0` (Ray / Tune / RLLib)](https://docs.ray.io/en/master/rllib.html) 
- This repository as a toolbox

To explore `RLLib` please refer to the 
[RLLib/Tune/Ray documentation](https://docs.ray.io/en/master/rllib-toc.html).
And also the [examples provided by RLLib](https://github.com/ray-project/ray/tree/master/rllib/examples). 

# Training models

There is 3 possible ways to run some training:

**Tune function API**  
The Tune function API where you only need to provide the training function. 
Best if you simply want to run some code from an external repository.
[See Tune documentation](https://docs.ray.io/en/master/tune/key-concepts.html).  
**Functionalities:** running several seeds in parallel and comparing their results, 
easy plotting to Tensorboard, visualizing the plots in live, 
saving configuration files / tracking your experiments, hyper-parameter search 

**The Tune class API**  
You need to provide a Trainer class with at minimum a setup method and a train method (one
 iteration one). Best if you want to run some code from an external repository and you need checkpoints or you 
want to evaluate (no training) it against other RLLib algorithms or with some experimentation tools in utils.
[See Tune documentation](https://docs.ray.io/en/master/tune/key-concepts.html).  
**Additional functionalities:** Checkpoints.

**RLLib Trainer class**  
You need to use RLLib Trainer and Policy classes. 
The RLLib Trainer class is the RLLib implementation for the Tune class API.
Best if you are creating a new training/policy from scratch.  
**Additional functionalities:** Using components from RLLib 
(models, environments, algorithms, exploration, etc.)


# Content
- envs
    - Matrix social dilemmas: `IteratedPrisonersDilemma`, `IteratedMatchingPennies`, 
    `IteratedStagHunt`, `IteratedChicken`, `BOTS_PD`
    - Coin games: `CoinGame`, `AsymCoinGame`
- algos
    - Inequity aversion: `InequityAversionTrainer` PyTorch only
    - L-TFT: 
        - `LE` (Learning Tit-for-tat or Learning Equilibrium) **Simplified version not working properly, WIP**
    - LOLA: 
        - `LOLA-PG` Official implementation running with `Tune`
        - `LOLA-DICE` Unofficial implementation **Not working properly, WIP**
- utils  
    - log: `stats_fn_wt_additionnal_logs`
  
**Custom examples:**

|          Env         |                 Algo                |             File            |               Status              |
|:--------------------:|:-----------------------------------:|:---------------------------:|:---------------------------------:|
|          IPD         |           Policy Gradient           |          [PG_IPD.py](https://github.com/Manuscrit/temp_research_framework/blob/master/examples/PG_IPD.py)           | Need to set good hyper-parameters |
| (Asymmetric) Coin game |                 PPO                 |     [PPO_AsymCoinGame.py](https://github.com/Manuscrit/temp_research_framework/blob/master/examples/PPO_AsymCoinGame.py)     | Need to set good hyper-parameters |
| (Asymmetric) Coin game |   LOLA-PG official implementation (with `Tune`)   |     [LOLA_PG_official.py](https://github.com/Manuscrit/temp_research_framework/blob/master/examples/tune_only/LOLA_PG_official.py)     | OK |
| (Asymmetric) Coin game |   LOLA-DICE official implementation (with `Tune`)   |     [LOLA_DICE_official.py](https://github.com/Manuscrit/temp_research_framework/blob/master/examples/tune_only/LOLA_DICE_official.py)     | Not cooperating, WIP |
|          IPD         | LOLA-DICE unofficial implementation |   [LOLA_DICE_unofficial.py](https://github.com/Manuscrit/temp_research_framework/blob/master/examples/LOLA_DICE_unofficial.py)    |     Not working properly, WIP     |
|          IPD         |        Simplified L-TFT (LE)        |          [LE_IPD.py](https://github.com/Manuscrit/temp_research_framework/blob/master/examples/LE_IPD.py)          |     Not working properly, WIP     |
|       BOTS + PD      |          Inequity Aversion          | [InequityAversion_BOTS_PD.py](https://github.com/Manuscrit/temp_research_framework/blob/master/examples/InequityAversion_BOTS_PD.py)  | Need to set good hyper-parameters |


# Starting
### Quick intro to RLLib
[RLlib in 60 seconds](https://docs.ray.io/en/master/rllib.html)  

### Install

```bash
# Install env
conda create -n marltoolbox python=3.8.5
conda activate marltoolbox

# Install marltoolbox
conda install psutil
git clone https://github.com/Manuscrit/temp_research_framework.git
cd temp_research_framework
pip install -e .

# Check that RLLib is working
rllib train --run=PPO --env=CartPole-v0 --torch 

# Check that marltoolbox is working
python ./examples/PG_IPD.py

# Visualize logs
tensorboard --logdir ~/ray_results
# If working on GCP: forward the conenction from a Virtual Machine(VM) to your machine
gcloud compute ssh {replace-by-instance-name} --zone=us-central1-a -- -NfL 6006:localhost:6006

# Stuff below are optionals: 

# Install PyTorch with GPU
# Check cuda version
nvidia-smi
# Look at "CUDA Version: XX.X"
# With the right cuda version
conda install pytorch torchvision cudatoolkit=[cuda version like 10.2] -c pytorch
# Check installation & GPU available
python
    import torch
    torch.__version__
    torch.cuda.is_available()
    exit()

# Install Tensorflow
pip install --upgrade pip
pip install tensorflow
```


# Some usages:
0 - Fall back to `Tune` when using `RLLib` is too costly  
1 - Using components directly provided by `RLLib`  
2 - Using custom environments  
3 - Customizing existing algorithms  
4 - Using custom agents (custom Policy: acting and training logic)   
5 - Using custom dataflows (custom Trainer or Trainer's execution_plan)

#### 0) Fall back to `Tune` when using `RLLib` is too costly
Examples:  
- LOLA-PG (official) with Coin game: 
[tune_only/LOLA_PG_official.py](https://github.com/Manuscrit/temp_research_framework/blob/master/examples/tune_only/LOLA_PG_official.py)
- LOLA-DICE (official) with Coin game: 
[tune_only/LOLA_DICE_official.py](https://github.com/Manuscrit/temp_research_framework/blob/master/examples/tune_only/LOLA_DICE_official.py)

If the setup you want to train already has a training loop 
and if the cost to convert it into `RLLib` is too expensive,
then with minimum changes you can use `Tune`
to take advantage of the following functionalities:
- running several seeds in parallel and comparing their results
- easy plotting to tensorboard
- visualizing the plots in live
- saving configuration files / tracking your experiments
- hyper-parameter search

Using `Tune` alone remove most of the constraints imposed by `RLLib` over the training process.
Given a training function, you will need to do two changes:
- Input the hyper-parameters through a config dictionary. 
- At each time step of your future plots, 
call tune.report() with a dictionary of values to log.   

You can find more detail in the [introduction to `Tune`](https://docs.ray.io/en/master/tune/key-concepts.html).  
You can find [more examples using `Tune` here](https://docs.ray.io/en/master/tune/examples/index.html#tune-general-examples
).  

**When is the conversion cost to `RLLib` too high?**  
- If the algorithm has a complex unusual dataflow 
- If the algorithm has an unusual training process 
    - like LOLA: performing virtual opponent updates
    - like L-TFT(LE): nested algorithms
- If you don't need to change it
- If you don't plan to run it against algorithms in RLLib
- If you do not plan to work much with the algorithm. 
And thus, you do not want to invest time in the conversion to `RLLib`.
- Above points and you are only starting to use `RLLib`  
- ...

#### 1) Using components directly provided by `RLLib`
Examples:
- APEX_DDPG and the water world environment:
[rllib/examples/multi_agent_independent_learning.py](https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent_independent_learning.py)
- MADDPG and the two step game environment:
[rllib/examples/two_step_game.py](https://github.com/ray-project/ray/blob/master/rllib/examples/two_step_game.py)
- Policy Gradient (PG) and the rock paper scissors environment (function `run_same_policy`):
[rllib/examples/rock_paper_scissors_multiagent.py](https://github.com/ray-project/ray/blob/master/rllib/examples/rock_paper_scissors_multiagent.py)

#### 2) Using custom environments
Examples:  
- IPD environment: 
[PG_IPD.py](https://github.com/Manuscrit/temp_research_framework/blob/master/examples/PG_IPD.py)
- (Asymmetric) Coin Game: 
[PPO_AsymCoinGame.py](https://github.com/Manuscrit/temp_research_framework/blob/master/examples/PPO_AsymCoinGame.py)

#### 3) Customizing existing algorithms
Examples:  
- Customize Policy's postprocessing (before training) and Trainer: 
[InequityAversion_BOTS_PD.py](https://github.com/Manuscrit/temp_research_framework/blob/master/examples/InequityAversion_BOTS_PD.py)
- Train some Policies from the Trainer to access opponents models *(dirty RLLib hack)*:
[LOLA_DICE_unofficial.py](https://github.com/Manuscrit/temp_research_framework/blob/master/examples/LOLA_DICE_unofficial.py)
- Change the loss function of the Policy Gradient (PG) Policy:
[rllib/examples/rock_paper_scissors_multiagent.py](https://github.com/ray-project/ray/blob/master/rllib/examples/rock_paper_scissors_multiagent.py)  
(in the `run_with_custom_entropy_loss` function) 

#### 4) Using custom agents (custom Policy: acting and training logic)  
Examples:  
- Hardcoded random Policy:
[rllib/examples/multi_agent_custom_policy.py](https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent_custom_policy.py)
- Hardcoded fixed Policy:
[rllib/examples/rock_paper_scissors_multiagent.py](https://github.com/ray-project/ray/blob/master/rllib/examples/rock_paper_scissors_multiagent.py)  
(in the `run_heuristic_vs_learned` function)
- Custom Policy with nested Policies (PyTorch):
[LE_IPD.py](https://github.com/Manuscrit/temp_research_framework/blob/master/examples/LE_IPD.py)

#### 5) Using custom dataflows (custom Trainer or Trainer's execution_plan)
Examples:
- Training 2 different policies with 2 different Trainers (less complex, less sample efficient):
[rllib/examples/multi_agent_two_trainers.py](https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent_two_trainers.py)
- Training 2 different policies with a custom Trainer (more complex, more sample efficient):
[rllib/examples/two_trainer_workflow.py](https://github.com/ray-project/ray/blob/master/rllib/examples/two_trainer_workflow.py)


All examples provided by RLLib are
[here](https://github.com/ray-project/ray/tree/master/rllib/examples). 

# Wishlist
- See the google doc "Survey for MARL framework"
- See the google doc "MARL benchmark paper"
- add unit tests
- add a simple coin game env (not vectorized and more readable)
- check performance of coin game with and without reporting the additional info
- set good hyper-parameters for our custom examples 
    - like 2nd conv without padding for Coin Game?
- RLLib install helper
- report all results directly in Weights&Biases (saving download time from VM)
- MARL RLLib tutorial
- give some guidelines for this toolbox